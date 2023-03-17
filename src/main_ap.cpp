#include "../include/mavlink/mavlink_types.h"
#include "eqvio/mathematical/IMUVelocity.h"
#include "eqvio/VIOFilter.h"
#include "eqvio/VIOFilterSettings.h"
#include "eqvio/mathematical/VisionMeasurement.h"
#include "eqvio/dataStream.h"
#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"
#include "opencv2/highgui/highgui.hpp"
#include "yaml-cpp/yaml.h"
#include "eqvio/LoopTimer.h"

static int dev_fd = -1;

static double get_time_seconds()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (tp.tv_sec + (tp.tv_usec*1.0e-6));
}

void comm_send_ch(mavlink_channel_t chan, uint8_t c)
{
    write(dev_fd, &c, 1);
}

// Open up a serial port at given baudrate
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

int main(int argc, const char *argv[])
{
    
    // Read arguments
    if (argc < 3)
    {
        std::cout<< "Not enough arguments were provided.\n";
        std::cout<< "Usage: EQVIO_config_file Serial_port_id"<<std::endl;
        return 1;
    }
    else if (argc > 3)
    {
        std::cout<< "Too many files were provided.\n";
        std::cout<< "Usage: EQVIO_config_file Serial_port_id"<<std::endl;
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

    loopTimer.initialise({"correction", "features", "preprocessing", "propagation", "total", "total vision update", "write output"});

    //find the camera intrensics file and configure the camera pointer
    const std::string cameraFileName = EQVIO_config_fname.substr(0, EQVIO_config_fname.rfind("/")) + "/undistort.yaml";
    assert(std::filesystem::exists(cameraFileName));

    cv::FileStorage fs(cameraFileName, cv::FileStorage::READ);
    cv::Mat K, dist;

    fs["camera_matrix"] >> K;
    fs["dist_coeffs"] >> dist;

    std::array<double, 4> distVec;
    for (int i = 0; i < dist.cols; ++i) {
        distVec[i] = dist.at<double>(i);
    }

    // Initialize the feature tracker and the filter
    dataStream ds;

    ds.camera = std::make_shared<GIFT::EquidistantCamera>(GIFT::EquidistantCamera(cv::Size(0, 0), K, distVec));

    VIOFilter::Settings filterSettings(eqf_vioConfig["eqf"]);
    ds.filter = VIOFilter(filterSettings);
    std::cout << "Filter initialized." << std::endl;

    ds.featureTracker = GIFT::PointFeatureTracker(ds.camera);
    ds.featureTracker.settings.configure(eqf_vioConfig["GIFT"]);

    //This checks to see if the default values for our camera parameters are being changed and changes them if required. 
    if (eqf_vioConfig["main"]["indoorLighting"])
    {
        ds.indoor_lighting = eqf_vioConfig["main"]["indoorLighting"].as<bool>();
    }
    if (eqf_vioConfig["main"]["cameraXResolution"])
    {
        ds.cameraXResolution = eqf_vioConfig["main"]["cameraXResolution"].as<int>();
    }
    if (eqf_vioConfig["main"]["cameraYResolution"])
    {
        ds.cameraYResolution = eqf_vioConfig["main"]["cameraYResolution"].as<int>();
    }
    if (eqf_vioConfig["main"]["cameraFrameRate"])
    {
        ds.cameraFrameRate = eqf_vioConfig["main"]["cameraFrameRate"].as<int>();
    }
    if (eqf_vioConfig["main"]["saveImages"])
    {
        ds.save_images = eqf_vioConfig["main"]["saveImages"].as<bool>();
    }
    std::cout << "Feature tracker initialized." << std::endl;
    
    // Open the given serial port
    int opt;
    const char *devname = argv[2];

    dev_fd = open_serial(devname, 921600); // This baudrate is used for connection between Pi and CubeOrange
    if (dev_fd == -1) {
        printf("Failed to open %s\n", devname);
        exit(1);
    }

    ds.fd = dev_fd;

    // Start the threads
    ds.startThreads();

    return 0;
}
