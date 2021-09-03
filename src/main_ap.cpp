#include "../include/mavlink/mavlink_types.h"
#include "eqf_vio/IMUVelocity.h"
#include "eqf_vio/VIOFilter.h"
#include "eqf_vio/VIOFilterSettings.h"
#include "eqf_vio/VisionMeasurement.h"
#include "eqf_vio/dataStream.h"
#include "GIFT/PointFeatureTracker.h"
#include "GIFT/Visualisation.h"
#include "opencv2/highgui/highgui.hpp"
#include "yaml-cpp/yaml.h"

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

    // Initialize the feature tracker and the filter
    const std::string camera_intrinsics_fname = eqf_vioConfig["GIFT"]["intrinsicsFile"].as<std::string>();
    if (!std::ifstream(camera_intrinsics_fname).good())
    {
        std::stringstream ess;
        ess << "Couldn't open the GIFT camera intrinsics file: "<< camera_intrinsics_fname;
        throw std::runtime_error(ess.str());
    }
    GIFT::PinholeCamera camera = GIFT::PinholeCamera(cv::String(camera_intrinsics_fname));    

    dataStream ds;

    // Todo: Hardcode the cam params --- to be fixed
    float k[9] = {550.2499495823959, 0.0, 634.970638005679, 0.0, 548.8753588860187, 381.1055873002101, 0.0, 0.0, 1.0}; 
    float d[4] = {-0.03584706281933589, 0.0077362868057236946,-0.04587986231938219, 0.04834004050933801};
    ds.K_coef = cv::Mat(3, 3, CV_32F, k);
    ds.D_coef = cv::Mat(1, 4, CV_32F, d);
    
    VIOFilter::Settings filterSettings(eqf_vioConfig["eqf"]);

    ds.filter = VIOFilter(filterSettings);
    ds.featureTracker = GIFT::PointFeatureTracker(camera);

    // safeConfig(eqf_vioConfig["GIFT"]["maxFeatures"], ds.featureTracker.maxFeatures);
    // safeConfig(eqf_vioConfig["GIFT"]["featureDist"], ds.featureTracker.featureDist);
    // safeConfig(eqf_vioConfig["GIFT"]["minHarrisQuality"], ds.featureTracker.minHarrisQuality);
    // safeConfig(eqf_vioConfig["GIFT"]["featureSearchThreshold"], ds.featureTracker.featureSearchThreshold);
    // safeConfig(eqf_vioConfig["GIFT"]["maxError"], ds.featureTracker.maxError);
    // safeConfig(eqf_vioConfig["GIFT"]["winSize"], ds.featureTracker.winSize);
    // safeConfig(eqf_vioConfig["GIFT"]["maxLevel"], ds.featureTracker.maxLevel);
     ds.featureTracker.settings.configure(eqf_vioConfig["GIFT"]);
    
    ds.indoor_lighting = eqf_vioConfig["main"]["indoorLighting"].as<bool>();
    
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
