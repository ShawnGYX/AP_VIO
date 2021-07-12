#include "eqf_vio/dataStream.h"

static mavlink_status_t mav_status;
static uint8_t target_system;
static double last_msg_s;
static double last_hb_s;

float gyro_factor = 1e-3;
float acc_factor = 9.81 * 1e-3;

cv::Mat record_cam(bool indoor_lighting);
VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp);

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
            if (msg.msgid == MAVLINK_MSG_ID_ATTITUDE) {
                printf("msgid=%u dt=%f\n", msg.msgid, tnow - last_msg_s);
                last_msg_s = tnow;

                mavlink_msg_raw_imu_decode(&msg, &raw_imu);
                imuVel.stamp = tnow;
                imuVel.omega << raw_imu.xgyro*gyro_factor, raw_imu.ygyro*gyro_factor, raw_imu.zgyro*gyro_factor;
                imuVel.accel << raw_imu.xacc*acc_factor, raw_imu.yacc*acc_factor, raw_imu.zacc*acc_factor;

                this->filter.processIMUData(imuVel);

            }
            if (target_system == 0 && msg.msgid == MAVLINK_MSG_ID_HEARTBEAT) {
                printf("Got system ID %u\n", msg.sysid);
                target_system = msg.sysid;

                // get key messages at 200Hz
                mav_set_message_rate(MAVLINK_MSG_ID_ATTITUDE, 40);
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
    std::cout<<"Image Message Received."<<std::endl;
    double now = get_time_seconds();
    
    // Run GIFT on the image 
    featureTracker.processImage(image);
    const std::vector<GIFT::Feature> features = featureTracker.outputFeatures();
    const VisionMeasurement visionData = convertGIFTFeatures(features, now);

    // Pass the feature data to the filter
    filter.processVisionData(visionData);

    // Request the system state from the filter
    VIOState estimatedState = filter.stateEstimate();
    return estimatedState;
}



// Start the IMU receiver and Camera capture threads
void dataStream::startThreads()
{

    std::time_t t0 = std::time(nullptr);
    std::stringstream outputFileNameStream;
    outputFileNameStream << "EQF_VIO_output_" << std::put_time(std::localtime(&t0), "%F_%T") << ".csv";
    
    
    outputFile = std::ofstream(outputFileNameStream.str());
    outputFile << "time, tx, ty, tz, qw, qx, qy, qz, vx, vy, vz, N, "
                << "p1id, p1x, p1y, p1z, ..., ..., ..., ..., pNid, pNx, pNy, pNz" << std::endl;
    
    recv_th = std::thread(&dataStream::recv_thread, this);
    printf("IMU Receiver thread created.\n");

    cam_th = std::thread(&dataStream::cam_thread, this);
    printf("Camera capture thread created.\n");

}

// Kill the threads
void dataStream::stopThreads()
{
    stop_recv = true;
    stop_cam = true;
    // stop_send = true;
    usleep(100);
    if (recv_th.joinable()){recv_th.join();}
    if (cam_th.joinable()){cam_th.join();}
    printf("Threads stopped.\n");
}


void dataStream::cam_thread()
{
    const cv::Mat& img_msg = record_cam(false);
    const VIOState stateEstimate = callbackImage(img_msg);

    outputFile << std::setprecision(20) << filter.getTime() << std::setprecision(5) << ", "
                               << stateEstimate << std::endl;

}


cv::Mat record_cam(bool indoor_lighting)
{
    // Initialize image capture module
    cv::VideoCapture *cap;
    cap = new cv::VideoCapture(0);
    cap->set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);

    // Adjust exposure
    float exposure;
    cv::Mat frame;
    if (indoor_lighting)
    {
        exposure = 0.5;
    }
    else
    {
        exposure = 0.001;
    }
    float gain = 1e-4;
    for(;;)
    {
        cap->read(frame);
        if (frame.empty())
        {
            std::cerr << "Blank frame captured!\n";
            break;
        }

        // Set camera exposure
        cap->set(cv::CAP_PROP_EXPOSURE, exposure);

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

    return frame;
    
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

