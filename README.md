# AP_VIO

This repository contains an implementation of an Equivariant Filter (EqF) for Visual Inertial Odometry (VIO), on the ArduPilot platform.

## Required Dependencies

- C++17 (GCC Version >=8.3)
- CMake ( for build environment ): ```sudo apt install cmake```

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
make -j2
```

### Known Issue

- When running ```cmake ..``` , CMake can't locate GIFT headers and returns an error. This can be fixed by manually setting the GIFT directory in ```CMakelist.txt``` . In line 14, change the address in the quotes to the one where you build GIFT on your device (e.g ```/home/GIFT/build/GIFT```).

## Preparations

### Camera Calibration 

### Camera-IMU Calibration

### Configuration File



## Usage Instructions

### Start VIO

To start VIO on the onboard computer, go to AP_VIO folder, execute the following command in the shell

```bash
./build/vio_ap <config_file> <serial port>
```

for example: ``` ./build/vio_ap EQVIO_config.yaml /dev/serial0```.


