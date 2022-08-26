#include "eqf_vio/dataStream.h"
#include <filesystem>
namespace fs = std::filesystem;

// Used for store frame messages
struct cam_msg
{
    cam_msg(double t, cv::Mat i) : t_now(t), img(i) {}
    double t_now;
    cv::Mat img;
};

struct mav_imu_message
{
    IMUVelocity imuVel;
    mavlink_gps_raw_int_t GPS_RAW;
    mavlink_global_position_int_t Global_pos;
    mavlink_attitude_t att;
};


std::mutex mtx_filter;
std::mutex mtx_cam_queue;
std::mutex mtx_imu_queue;
std::mutex mtx_cam_save_queue;

// Passing messages between recv and proc threads
std::queue<cam_msg> cam_queue;
std::queue<IMUVelocity> imu_queue;
std::queue<cam_msg> cam_save_queue;

static mavlink_status_t mav_status;
static uint8_t target_system;
static double last_msg_s;
static double last_msg_s_cam;

static double last_hb_s;

float gyro_factor = 1e-3;
float acc_factor = 9.81 * 1e-3;

void mav_set_message_rate(uint32_t message_id, float rate_hz)
{
    mavlink_msg_command_long_send(MAVLINK_COMM_0,
                                  target_system,
                                  0,
                                  MAV_CMD_SET_MESSAGE_INTERVAL,
                                  0,
                                  message_id,
                                  (uint32_t)(1e6 / rate_hz),
                                  0, 0, 0, 0, 0);
}

void send_heartbeat(void)
{
    printf("sending hb\n");
    mavlink_msg_heartbeat_send(
        MAVLINK_COMM_0,
        MAV_TYPE_GCS,
        MAV_AUTOPILOT_GENERIC,
        0, 0, 0);
}

static double get_time_seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (tp.tv_sec + (tp.tv_usec * 1.0e-6));
}

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature> &GIFTFeatures, const double &stamp)
{
    VisionMeasurement measurement;
    measurement.stamp = stamp;
    measurement.numberOfBearings = GIFTFeatures.size();
    measurement.bearings.resize(GIFTFeatures.size());
    std::transform(GIFTFeatures.begin(), GIFTFeatures.end(), measurement.bearings.begin(), [](const GIFT::Feature &f)
                   {
                       Point3d pt;
                       pt.p = f.sphereCoordinates();
                       pt.id = f.idNumber;
                       return pt;
                   });
    return measurement;
}

// Kill the threads
dataStream::~dataStream()
{
    try
    {
        stopThreads();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

// Callback function for images
call_back_img_returned_values dataStream::callbackImage(const cv::Mat image, const double ts)
{
    //define the structure we will return to the main thread 
    call_back_img_returned_values returned_values; 
    //we lock the filter to the imu measurments to avoid having imu measurments with later time stamp being processed before the image 
    mtx_filter.lock();
    // Undistort the image
    cv::Mat undistorted = image.clone();
    cv::Size imageSize = image.size();
    cv::Mat mapX = cv::Mat(imageSize, CV_32FC1);
    cv::Mat mapY = cv::Mat(imageSize, CV_32FC1);
    cv::Mat iD = cv::Mat::eye(3, 3, CV_32F);
    cv::fisheye::initUndistortRectifyMap(K_coef, D_coef, iD, K_coef, imageSize, CV_32FC1, mapX, mapY);
    cv::remap(image, undistorted, mapX, mapY, cv::INTER_LINEAR);

    // Run GIFT on the undistorted image
    featureTracker.processImage(undistorted);
    const std::vector<GIFT::Feature> features = featureTracker.outputFeatures();
    std::cout << "New image received, with " << features.size() << " features." << std::endl;
    returned_values.visionData = convertGIFTFeatures(features, ts);

    // Pass the feature data to the filter
    filter.processVisionData(returned_values.visionData);
    // Request the system state from the filter
    returned_values.estimatedState = filter.stateEstimate();
    mtx_filter.unlock();

    return returned_values;
}

// Start the IMU receiver and Camera capture threads
void dataStream::startThreads()
{
    // setup the output directory
    std::time_t t0 = std::time(nullptr);
    //Create a directory to store our output files 
    outputFolderStream << "EQVIO_output_" << (std::put_time(std::localtime(&t0), "%F_%T")) << "/";
    fs::create_directory(outputFolderStream.str());

    //setup the output writter for the apvio
    std::stringstream apvioOutputStream;
    apvioOutputStream << "EQVIO_output_" << (std::put_time(std::localtime(&t0), "%F_%T")) << "/" << "apvio_output/";
    vioWriter(outputFolderStream.str());

    // Set up recording file
    std::stringstream mav_imu_NameStream;
    mav_imu_NameStream << outputFolderStream.str() << "mav_imu.csv";
    mav_imu = std::ofstream(mav_imu_NameStream.str());
    mav_imu << "Timestamp (s), xgyro (rad/s), ygyro (rad/s), zgyro (rad/s), xacc (m/s/s), yacc (m/s/s), zacc (m/s/s), Mav Time (s), "
               << "Roll (rad), Pitch (rad), Yaw (rad), Lat (deg), Lon (deg), Alt (m), GLat (deg), GLon (deg), GAlt (m)" << std::endl;

    std::stringstream cam_NameStream;
    cam_NameStream << outputFolderStream.str() << "cam.csv";
    cam = std::ofstream(cam_NameStream.str());
    cam << "Timestamp (s), frame_num" << std::endl;

    // Start the threads
    imu_recv_th = std::thread(&dataStream::imu_recv_thread, this);
    // printf("Start receiving IMU message.\n");
    imu_proc_th = std::thread(&dataStream::imu_proc_thread, this);
    // printf("Start processing IMU message.\n");
    cam_recv_th = std::thread(&dataStream::cam_recv_thread, this);
    // printf("Start receiving camera frames.\n");
    cam_proc_th = std::thread(&dataStream::cam_proc_thread, this);
    // printf("Start processing camera frames.\n");
    cam_save_th = std::thread(&dataStream::cam_save_thread, this);
    printf("All threads initialized.\n");
}

// Kill the threads
void dataStream::stopThreads()
{
    usleep(100);
    if (imu_recv_th.joinable())
    {
        imu_recv_th.join();
    }
    if (cam_recv_th.joinable())
    {
        cam_recv_th.join();
    }
    if (imu_proc_th.joinable())
    {
        imu_proc_th.join();
    }
    if (cam_proc_th.joinable())
    {
        cam_proc_th.join();
    }
    printf("Threads stopped.\n");
}

void dataStream::imu_recv_thread()
{
    uint8_t b;
    mavlink_message_t msg;
    mavlink_raw_imu_t raw_imu;
    IMUVelocity imuVel;
    mav_imu_message mavData;
    bool att_flag, global_gps_flag, raw_gps_flag = false;

    while (read(fd, &b, 1) == 1)
    {
        if (mavlink_parse_char(MAVLINK_COMM_0, b, &msg, &mav_status))
        {
            double tnow = get_time_seconds();
            //go through and check each message ID. then unpack the data and save it to the mavData structure 
            if (msg.msgid == MAVLINK_MSG_ID_ATTITUDE)
            {
                mavlink_msg_attitude_decode(&msg, &mavData.att);
                att_flag = true;
            }
            else if (msg.msgid == MAVLINK_MSG_ID_GLOBAL_POSITION_INT)
            {
                mavlink_msg_global_position_int_decode(&msg, &mavData.Global_pos);
                global_gps_flag = true;
            }
            else if (msg.msgid == MAVLINK_MSG_ID_GPS_RAW_INT)
            {
                mavlink_msg_gps_raw_int_decode(&msg, &mavData.GPS_RAW);
                raw_gps_flag = true;
            }
            //check to see if all fields of mavData have data in them
            else if ((msg.msgid == MAVLINK_MSG_ID_RAW_IMU) and (att_flag) and (global_gps_flag) and (raw_gps_flag))
            {
                last_msg_s = tnow;

                // Decode the mavlink msg and store in the filter format
                mavlink_msg_raw_imu_decode(&msg, &raw_imu);
                imuVel.stamp = tnow;
                imuVel.omega << raw_imu.xgyro*gyro_factor, raw_imu.ygyro*gyro_factor, raw_imu.zgyro*gyro_factor;
                imuVel.accel << raw_imu.xacc*acc_factor, raw_imu.yacc*acc_factor, raw_imu.zacc*acc_factor;

                mavData.imuVel = imuVel;

                //write to our csv with the most up to data data
                mav_imu << std::setprecision(20) << tnow << std::setprecision(5) << ", ";
                mav_imu << mavData.imuVel.omega.x() << "," << mavData.imuVel.omega.y() << "," << mavData.imuVel.omega.z() << ","; 
                mav_imu << mavData.imuVel.accel.x() << "," << mavData.imuVel.accel.y() << "," << mavData.imuVel.accel.z() << ","; 
                mav_imu << raw_imu.time_usec*1e6 << ",";
                mav_imu << mavData.att.roll << "," << mavData.att.pitch << "," << mavData.att.yaw << ",";
                mav_imu << mavData.Global_pos.lat*1e-7 << "," << mavData.Global_pos.lon*1e-7 << "," << mavData.Global_pos.alt*1e-3 << ",";
                mav_imu << mavData.GPS_RAW.lat*1e-7 << "," << mavData.GPS_RAW.lon*1e-7 << "," << mavData.GPS_RAW.alt*1e-3 << "," << std::endl;

                // Push the message to the queue.
                // The maximum size of the queue is 10.
                mtx_imu_queue.lock();
                imu_queue.push(imuVel);
                mtx_imu_queue.unlock();
                if (imu_queue.size() > 10)
                {
                    imu_queue.pop();
                }
            }
            if (target_system == 0 && msg.msgid == MAVLINK_MSG_ID_HEARTBEAT)
            {
                printf("Got system ID %u\n", msg.sysid);
                target_system = msg.sysid;
                // Request IMU messages at 200Hz
                mav_set_message_rate(MAVLINK_MSG_ID_RAW_IMU, 200);
                // request ground truth data at 20Hz
                mav_set_message_rate(MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 20);
                mav_set_message_rate(MAVLINK_MSG_ID_GPS_RAW_INT, 20);
                mav_set_message_rate(MAVLINK_MSG_ID_ATTITUDE, 20);
            }
        }
    }
    double tnow = get_time_seconds();
    if (tnow - last_hb_s > 1.0)
    {
        last_hb_s = tnow;
        send_heartbeat();
    }
}

void dataStream::cam_recv_thread()
{
    // Start video capture, disable auto exposure tuning.
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
    cap.set(cv::CAP_PROP_EXPOSURE, 1.7);
    cv::Mat frame;
    float exposure;  
    if ( indoor_lighting ){ exposure =0.3; } else{ exposure = 0.001; }

    float gain = 1e-4;

    for (;;)
    {
        double t1 = get_time_seconds();
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "Something is wrong with the webcam, could not get frame." << std::endl;
            break;
        }

        // Add new message to the queue. The size limit is 2.
        mtx_cam_queue.lock();
        cam_queue.push(cam_msg(t1, frame));
        mtx_cam_queue.unlock();
        if (cam_queue.size() > 10)
        {
            cam_queue.pop();
        }
        mtx_cam_save_queue.lock();
        cam_save_queue.push(cam_msg(t1,frame));
        mtx_cam_save_queue.unlock();
        if (cam_save_queue.size() >10)
        {
            cam_save_queue.pop();
        }


        // Adjust camera exposure
        cap.set(cv::CAP_PROP_EXPOSURE, exposure);
        cv::Scalar img_mean_s = cv::mean(frame);
        float img_mean = img_mean_s[0];
        if (img_mean > 128 - 32 && img_mean < 128 + 32)
        {
            continue;
        }
        exposure += gain * (128 - img_mean) * exposure;
        if (exposure > 0.7)
        {
            exposure = 0.7;
        }
        else if (exposure <= 0.0)
        {
            exposure = 1e-6;
        }
    }
}

void dataStream::cam_proc_thread()
{
    while (true)
    {
        if (!cam_queue.empty())
        {
            mtx_cam_queue.lock();
            cam_msg tobeProc = cam_queue.back();
            mtx_cam_queue.unlock();
            call_back_img_returned_values processed_filtered_values = callbackImage(tobeProc.img, tobeProc.t_now);

            vioWriter.writeStates(filter.getTime(), processed_filtered_values.estimatedState);
            vioWriter.writeFeatures(processed_filtered_values.visionData);

            // Send VP message to AutoPilot
            update_vp_estimate(processed_filtered_values.estimatedState);

        }
        usleep(100);
    }
}

void dataStream::cam_save_thread()
{
    int image_number = 0;
    while (true)
    {
        if (!cam_save_queue.empty())
        {
            mtx_cam_save_queue.lock();
            cam_msg tobeSave = cam_save_queue.back();
            mtx_cam_save_queue.unlock(); 
            //create the image name in the correct directory
            std::string dir(outputFolderStream.str()+"frames/frame_"+std::to_string(image_number)+".jpg");
            //write the image 
            cv::imwrite(dir, tobeSave.img);
            //write to cam.csv
            cam << tobeSave.t_now << "," << image_number << "," << std::endl;
            image_number++;
        }
        usleep(100);
    }
}


void dataStream::imu_proc_thread()
{
    while (true)
    {
        // If there's message in the queue, read the last one and pass into the filter.
        if (!imu_queue.empty())
        {
            mtx_imu_queue.lock();
            IMUVelocity tobeProc = imu_queue.back();
            mtx_imu_queue.unlock();
            mtx_filter.lock();
            this->filter.processIMUData(tobeProc);
            mtx_filter.unlock();
        }
        usleep(100);
    }
}

bool dataStream::should_send(TypeMask type_mask) const
{
    return 1;
}

void dataStream::update_vp_estimate(const VIOState estimatedState)
{
    const uint64_t now_us = get_time_seconds() * 1e6;

    // Calculate a random time offset to the time sent in the message
    if (time_offset_us == 0)
    {
        time_offset_us = (unsigned(random()) % 7000) * 1000000ULL;
        printf("time_off_us %llu\n", (long long unsigned)time_offset_us);
    }

    if (now_us - last_observation_usec < 20000)
    {
        return;
    }

    // Load estimates from the filter
    const Eigen::Quaterniond &attitude = estimatedState.pose.R().asQuaternion();
    const Eigen::Vector3d &position = estimatedState.pose.x();
    const Eigen::Vector3d &vel = estimatedState.velocity;

    Eigen::Vector3d euler = attitude.toRotationMatrix().eulerAngles(0, 1, 2);
    float roll = euler[0];
    float pitch = euler[1];
    float yaw = euler[2];

    // Send message
    uint32_t delay_ms = 25 + unsigned(random()) % 100;
    uint64_t time_send_us = now_us + delay_ms * 1000UL;

    if (should_send(TypeMask::VISION_POSITION_ESTIMATE))
    {
        mavlink_msg_vision_position_estimate_send(
            MAVLINK_COMM_0,
            now_us + time_offset_us,
            position.x(),
            position.y(),
            position.z(),
            roll,
            pitch,
            yaw,
            NULL, 0);
    }

    if (should_send(TypeMask::VISION_SPEED_ESTIMATE))
    {
        mavlink_msg_vision_speed_estimate_send(
            MAVLINK_COMM_0,
            now_us + time_offset_us,
            vel.x(),
            vel.y(),
            vel.z(),
            NULL, 0);
    }
}