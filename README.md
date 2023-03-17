# AP_VIO

This repository contains an implementation of an Equivariant Filter (EqF) for Visual Inertial Odometry (VIO), on the ArduPilot platform.

## Required Dependencies

- C++17 (GCC Version >=10.1)
- CMake ( for build environment, Version >=3.12 ): ```sudo apt install cmake```

- Eigen 3: ```sudo apt install libeigen3-dev```

- Yaml-cpp: ```sudo apt install libyaml-cpp-dev```

- GIFT ( for feature tracking ): ```https://github.com/pvangoor/GIFT```

## Required Hardware Setup

- Onboard computer (e.g Rapberry Pi)
- Global shutter USB camera (needs to be hard mounted)
- Autopilot instance with IMUs (e.g Cube Orange)

## Building

To checkout and build AP_VIO on the Pi, execute the following at a shell:

```bash
git clone https://github.com/ShawnGYX/AP_VIO.git
cd AP_VIO
mkdir build
cd build
cmake ..
make 
```

### Known Issue

- There is a known issue with data logging due to conflict between mutex locks in aofstream.h

## Preparations

### Camera Calibration 
We use a checkerboard to calibrate the camera. Download the tools from ```https://github.com/STR-ANU/VIO-Tools```.

### Camera-IMU Calibration
We use Kalibr to calibrate camera to IMU. Available at: https://github.com/ethz-asl/kalibr. Can also be used for Camera calibration

## Usage Instructions

### Start VIO

To start VIO on the onboard computer, go to AP_VIO folder, execute the following command in the shell

```bash
./build/vio_ap <config_file> <serial port>
```

for example: ``` ./build/vio_ap EQVIO_config_simple.yaml /dev/serial0```.


