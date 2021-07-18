#pragma once



#include "CSVReader.h"
#include "IMUVelocity.h"
#include "VIOFilter.h"
#include "VIOFilterSettings.h"
#include "VisionMeasurement.h"

// #include "serial_port.h"


#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include "yaml-cpp/yaml.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <exception>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <signal.h>

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include <mutex>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#define MAVLINK_USE_CONVENIENCE_FUNCTIONS

#include "../include/mavlink/mavlink_types.h"
static mavlink_system_t mavlink_system = {42,11,};

extern void comm_send_ch(mavlink_channel_t chan, uint8_t c);

#include "../include/mavlink/ardupilotmega/mavlink.h"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

class dataStream{
    public:
    dataStream();
    ~dataStream();
    void startThreads();
    void stopThreads();
    void cam_thread();
    void recv_thread();
    void send_thread();
    void update_vp_estimate(const VIOState estimatedState);
    std::ofstream outputFile;
    bool indoor_lighting;
    int fd;

    cv::Mat K_coef;
    cv::Mat D_coef;

    GIFT::PointFeatureTracker featureTracker;
    VIOFilter filter;

    VIOState callbackImage(const cv::Mat image);

    VIOState tobeSend;

    private:

    std::thread recv_th;
    std::thread cam_th;
    std::thread send_th;
    bool stop_recv = false;
    bool stop_cam = false;
    bool stop_send = false;
    bool send_ready = false;
    uint64_t last_observation_usec;
    uint64_t time_offset_us;

    enum class TypeMask: uint8_t {
        VISION_POSITION_ESTIMATE   = (1 << 0),
        VISION_SPEED_ESTIMATE      = (1 << 1),
        VISION_POSITION_DELTA      = (1 << 2)
    };

    bool should_send(TypeMask type_mask) const;
    uint32_t last_heartbeat_ms;
};

