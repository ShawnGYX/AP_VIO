cmake_minimum_required(VERSION 3.0)
project(AP_VIO VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)

include(CMakeDependentOption)

find_package(Eigen3 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Set your own GIFT directory, usually in '../GIFT/build/GIFT'
set(GIFT_DIR "/home/shawn/Projects//GIFT/build/GIFT")


set(DEFAULT_CONFIG_FILE "EQF_VIO_config_template.yaml")
get_filename_component(DEFAULT_CONFIG_FILE ${DEFAULT_CONFIG_FILE} ABSOLUTE)
add_compile_definitions(DEFAULT_CONFIG_FILE="${DEFAULT_CONFIG_FILE}")

option( USE_MARCH_NATIVE "Use the flag -march=native" OFF)

if (USE_MARCH_NATIVE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    else()
        message("march=native was requested but is not supported.")
    endif()
endif()

add_subdirectory(libs)

set(AP_VIO_HEADER_FILES
    include/eqf_vio/VIOState.h
    include/eqf_vio/IMUVelocity.h
    include/eqf_vio/VIOGroup.h
    include/eqf_vio/VIOFilter.h
    include/eqf_vio/VIOFilterSettings.h
    include/eqf_vio/CSVReader.h
    include/eqf_vio/VisionMeasurement.h
    include/eqf_vio/EqFMatrices.h
    include/eqf_vio/dataStream.h
    )
set(AP_VIO_SOURCE_FILES
    src/VIOState.cpp
    src/VIOGroup.cpp
    src/VIOFilter.cpp
    src/VisionMeasurement.cpp
    src/IMUVelocity.cpp
    src/EqFMatrices.cpp
    src/dataStream.cpp
)

add_library(ap_vio_lib
    ${AP_VIO_HEADER_FILES}
    ${AP_VIO_SOURCE_FILES}
)

message("This is being included!")
message(${MODULE_INCLUDE_DIRS})

target_include_directories(ap_vio_lib
    PUBLIC include
    PUBLIC ${MODULE_INCLUDE_DIRS}
    PUBLIC ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(ap_vio_lib
    ${MODULE_LIBS}
    ${OpenCV_LIBS}
)


find_package(OpenCV REQUIRED)
find_package(GIFT REQUIRED)

include_directories ("/usr/include/opencv4/")

add_executable(vio_ap src/main_ap.cpp)

target_link_libraries(vio_ap
    GIFT
    ${OpenCV_LIBS}
    ap_vio_lib
    ${MODULE_LIBS}
    yaml-cpp
)
target_include_directories(vio_ap
    PRIVATE GIFT::GIFT
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE include
    PRIVATE ${EIGEN_INCLUDE_DIRS}
    PRIVATE ${YAML_CPP_INCLUDE_DIR}
)

