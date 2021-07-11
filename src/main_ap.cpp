#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <sys/ioctl.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <asm/ioctls.h>
#include <asm/termbits.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <thread>
#include <exception>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#include "mavhelper.h"

#define MAVLINK_USE_CONVENIENCE_FUNCTIONS

#include "../include/mavlink/mavlink_types.h"
mavlink_system_t mavlink_system = {42,11,};

static int dev_fd = -1;


#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VIOFilter.h"
#include "eqf_vio/VIOFilterSettings.h"
#include "eqf_vio/VisionMeasurement.h"
#include "eqf_vio/dataStream.h"
#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"
#include "opencv2/highgui/highgui.hpp"

#include "yaml-cpp/yaml.h"

static double get_time_seconds()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (tp.tv_sec + (tp.tv_usec*1.0e-6));
}

// Record the images
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


void comm_send_ch(mavlink_channel_t chan, uint8_t c)
{
    write(dev_fd, &c, 1);
}

/*
  open up a serial port at given baudrate
 */
static int open_serial(const char *devname, uint32_t baudrate)
{
    int fd = open(devname, O_RDWR|O_CLOEXEC);
    if (fd == -1) {
        return -1;
    }

    struct termios2 tc;
    memset(&tc, 0, sizeof(tc));
    if (ioctl(fd, TCGETS2, &tc) == -1) {
        return -1;
    }
    
    /* speed is configured by c_[io]speed */
    tc.c_cflag &= ~CBAUD;
    tc.c_cflag |= BOTHER;
    tc.c_ispeed = baudrate;
    tc.c_ospeed = baudrate;

    tc.c_cflag &= ~(PARENB|CSTOPB|CSIZE);
    tc.c_cflag |= CS8;

    tc.c_lflag &= ~(ICANON|ECHO|ECHOE|ISIG);
    tc.c_iflag &= ~(IXON|IXOFF|IXANY);
    tc.c_oflag &= ~OPOST;
    
    if (ioctl(fd, TCSETS2, &tc) == -1) {
        return -1;
    }
    if (ioctl(fd, TCFLSH, TCIOFLUSH) == -1) {
        return -1;
    }

    return fd;
}

/*
  show usage
 */
static void usage(void)
{
    printf("mavexample: <options>\n");
    printf(" -D serial_device\n");
}




int main(int argc, const char *argv[])
{
    
    // Read argument
    if (argc != 1)
    {
        std::cout<< "No configuration file was provided.\n";
        std::cout<< "Usage: EQVIO_config_file."<<std::endl;
        return 1;
    }
    else if (argc > 1)
    {
        std::cout<< "Too many files were provided.\n";
        std::cout<< "Usage: EQVIO_config_file."<<std::endl;
        return 1;
    }
    std::string EQVIO_config_fname(argv[1]);

    // Load EQVIO configurations
    if (!std::ifstream(EQVIO_config_fname).good())
    {
        std::stringstream ess;
        ess << "Couldn't open the configuration file: "<< EQVIO_config_fname;
        throw std::runtime_error(ess.str());
    }
    const YAML::Node eqf_vioConfig = YAML::LoadFile(EQVIO_config_fname);

    // Initialize the feature tracker and the filter
    const std::string camera_intrinsics_fname = eqf_vioConfig["GIFT"]["intrinsicsFile"].as<std::string>();
    if (!std::ifstream(camera_intrinsics_fname).good())
    {
        std::stringstream ess;
        ess << "Couldn't open the GIFT camera intrinsics file: "<< camera_intrinsics_fname;
        throw std::runtime_error(ess.str());
    }
    GIFT::PinholeCamera camera = GIFT::PinholeCamera(cv::String(camera_intrinsics_fname));
    GIFT::PointFeatureTracker featureTracker = GIFT::PointFeatureTracker(camera);
    // dataStream ds;
    VIOFilter::Settings filterSettings(eqf_vioConfig["eqf"]);
    VIOFilter filter = VIOFilter(filterSettings);

    // ds.filter = VIOFilter(filterSettings);
    // ds.featureTracker = GIFT::PointFeatureTracker(camera);

    safeConfig(eqf_vioConfig["GIFT"]["maxFeatures"], featureTracker.maxFeatures);
    safeConfig(eqf_vioConfig["GIFT"]["featureDist"], featureTracker.featureDist);
    safeConfig(eqf_vioConfig["GIFT"]["minHarrisQuality"], featureTracker.minHarrisQuality);
    safeConfig(eqf_vioConfig["GIFT"]["featureSearchThreshold"], featureTracker.featureSearchThreshold);
    safeConfig(eqf_vioConfig["GIFT"]["maxError"], featureTracker.maxError);
    safeConfig(eqf_vioConfig["GIFT"]["winSize"], featureTracker.winSize);
    safeConfig(eqf_vioConfig["GIFT"]["maxLevel"], featureTracker.maxLevel);
    
    
    int opt;
    const char *devname = NULL;

    while ((opt = getopt(argc, (char * const *)argv, "D:")) != -1) {
        switch (opt) {
        case 'D':
            devname = optarg;
            break;
        default:
            printf("Invalid option '%c'\n", opt);
            usage();
            exit(1);
        }
    }

    if (!devname) {
        usage();
        exit(1);
    }

    dev_fd = open_serial(devname, 115200);
    if (dev_fd == -1) {
        printf("Failed to open %s\n", devname);
        exit(1);
    }

    while (true) {
        // run at 100Hz
        usleep(1000*100);
        int ret = mav_update(dev_fd);
        update_vp_estimate();
        if (ret != 0) {
            printf("Failed mav_update\n");
            exit(1);
        }
    }

    return 0;
}
