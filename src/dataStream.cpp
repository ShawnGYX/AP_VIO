#include "eqf_vio/dataStream.h"

std::mutex mtx;
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
    gettimeofday(&tp,NULL);
    return (tp.tv_sec + (tp.tv_usec*1.0e-6));
}

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp)
{
    VisionMeasurement measurement;
    measurement.stamp = stamp;
    measurement.numberOfBearings = GIFTFeatures.size();
    measurement.bearings.resize(GIFTFeatures.size());
    std::transform(GIFTFeatures.begin(), GIFTFeatures.end(), measurement.bearings.begin(), [](const GIFT::Feature& f) {
        Point3d pt;
        pt.p = f.sphereCoordinates();
        pt.id = f.idNumber;
        return pt;
    });
    return measurement;
}

// Start the IMU receiver and Camera capture threads
dataStream::dataStream()
{

}

// Kill the threads
dataStream::~dataStream()
{
    try
    {
        stopThreads();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}

void dataStream::recv_thread()
{
    uint8_t b;
    mavlink_message_t msg;
    mavlink_raw_imu_t raw_imu;   
    IMUVelocity imuVel;

    while (read(fd, &b, 1) == 1) {
        if (mavlink_parse_char(MAVLINK_COMM_0, b, &msg, &mav_status)) {
            double tnow = get_time_seconds();
            if (msg.msgid == MAVLINK_MSG_ID_RAW_IMU) {
                // printf("msgid=%u dt=%f\n", msg.msgid, tnow - last_msg_s);
                last_msg_s = tnow;

                mavlink_msg_raw_imu_decode(&msg, &raw_imu);
                imuVel.stamp = tnow;
                imuVel.omega << raw_imu.xgyro*gyro_factor, raw_imu.ygyro*gyro_factor, raw_imu.zgyro*gyro_factor;
                imuVel.accel << raw_imu.xacc*acc_factor, raw_imu.yacc*acc_factor, raw_imu.zacc*acc_factor;

                // Pass the IMU measurements to the filter
                mtx.lock();
                this->filter.processIMUData(imuVel);
                mtx.unlock();
            }
            if (target_system == 0 && msg.msgid == MAVLINK_MSG_ID_HEARTBEAT) {
                printf("Got system ID %u\n", msg.sysid);
                target_system = msg.sysid;

                // get key messages at 200Hz
                mav_set_message_rate(MAVLINK_MSG_ID_RAW_IMU, 200);
            }
        }
    }

    double tnow = get_time_seconds();
    if (tnow - last_hb_s > 1.0) {
        last_hb_s = tnow;
        send_heartbeat();
    }
}



// Callback function for images
VIOState dataStream::callbackImage(const cv::Mat image)
{
    double now = get_time_seconds();

    // Undistort the image
    cv::Mat undistorted = image.clone();    
    cv::Size imageSize = image.size();
    cv::Mat mapX = cv::Mat(imageSize,CV_32FC1);
    cv::Mat mapY = cv::Mat(imageSize,CV_32FC1);
    cv::Mat iD = cv::Mat::eye(3,3,CV_32F);
    cv::fisheye::initUndistortRectifyMap(K_coef,D_coef,iD,K_coef,imageSize,CV_32FC1,mapX,mapY);
    cv::remap(image,undistorted,mapX, mapY, CV_INTER_LINEAR);

    // Run GIFT on the undistorted image 
    featureTracker.processImage(undistorted);
    const std::vector<GIFT::Feature> features = featureTracker.outputFeatures();
    std::cout<< "New image received, with" <<features.size()<<" features."<<std::endl;
    cv::imwrite("test.jpg", undistorted);
    const VisionMeasurement visionData = convertGIFTFeatures(features, now);

    // Pass the feature data to the filter
    mtx.lock();
    filter.processVisionData(visionData);
    // Request the system state from the filter
    VIOState estimatedState = filter.stateEstimate();
    mtx.unlock();
    send_ready = true;

    return estimatedState;
}



// Start the IMU receiver and Camera capture threads
void dataStream::startThreads()
{
    // Set output file, add header
    std::time_t t0 = std::time(nullptr);
    std::stringstream outputFileNameStream;
    outputFileNameStream << "EQF_VIO_output_" << std::put_time(std::localtime(&t0), "%F_%T") << ".csv";
    
    outputFile = std::ofstream(outputFileNameStream.str());
    outputFile << "time, tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, N, "
                << "p1id, p1x, p1y, p1z, ..., ..., ..., ..., pNid, pNx, pNy, pNz" << std::endl;
    
    // Start the threads
    recv_th = std::thread(&dataStream::recv_thread, this);
    printf("IMU Receiver thread created.\n");

    cam_th = std::thread(&dataStream::cam_thread, this);
    printf("Camera capture thread created.\n");

    send_th = std::thread(&dataStream::send_thread, this);
    printf("Sending thread created.\n");
}

// Kill the threads
void dataStream::stopThreads()
{
    stop_recv = true;
    stop_cam = true;
    stop_send = true;
    usleep(100);
    if (recv_th.joinable()){recv_th.join();}
    if (cam_th.joinable()){cam_th.join();}
    if (send_th.joinable()){send_th.join();};
    printf("Threads stopped.\n");
}


void dataStream::cam_thread()
{
    // Start video capture, disable auto exposure tuning.
    cv::VideoCapture cap(0);
    cap.set(CV_CAP_PROP_AUTO_EXPOSURE,0.25);
    cap.set(CV_CAP_PROP_EXPOSURE, 1.7);

    float exposure;
    cv::Mat frame;
    
    if (indoor_lighting)
    {
        exposure =0.5;
    }
    else
    {
        exposure = 0.001;
    }

    float gain = 1e-4;

    for(;;)
    {
        cap >> frame;
        if(frame.empty())
        {
            std::cerr << "Something is wrong with the webcam, could not get frame." << std::endl;
            break;
        }
        double tnow_cam = get_time_seconds();
        VIOState stateEstimate = callbackImage(frame);
        printf("cam dt=%f\n", tnow_cam - last_msg_s_cam);
        last_msg_s_cam = tnow_cam;

        // this needs to be atomic
        tobeSend = stateEstimate;

        outputFile << std::setprecision(20) << filter.getTime() << std::setprecision(5) << ", "
                               << stateEstimate << std::endl;

        // Manually adjust camera exposure
        cap.set(CV_CAP_PROP_EXPOSURE, exposure);

        cv::Scalar img_mean_s = cv::mean(frame);
        float img_mean = img_mean_s[0];
        if (img_mean > 128-32 && img_mean < 128+32)
        {
            continue;
        }
        exposure += gain * (128 - img_mean) * exposure;
        if (exposure > 0.7)
        {
            exposure = 0.7;
        }
        else if (exposure <=0.0)
        {
            exposure = 1e-6;
        }
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

    // load estimates from the filter
    const Eigen::Quaterniond& attitude = estimatedState.pose.R().asQuaternion();
    const Eigen::Vector3d& position = estimatedState.pose.x();
    const Eigen::Vector3d& vel = estimatedState.velocity;

    Eigen::Vector3d euler = attitude.toRotationMatrix().eulerAngles(0,1,2);
    float roll = euler[0];
    float pitch = euler[1];
    float yaw = euler[2];

    // send message
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
            NULL, 0
        );
    }
}


void dataStream::send_thread()
{
    while(true)
    {
        if (!send_ready)
        {
            continue;
        }
        else
        {
            update_vp_estimate(tobeSend);
            send_ready = false;
        }
        // Run at 10 Hz
        usleep(1000*100);
    }
}
