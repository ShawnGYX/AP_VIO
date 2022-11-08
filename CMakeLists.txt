cmake_minimum_required(VERSION 3.0)
project(AP_VIO VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER "/opt/gcc-10.1.0")
set(CMAKE_CXX_COMPILER "/opt/gcc-10.1.0/bin/g++-10.1")

include(CMakeDependentOption)

find_package(Eigen3 3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories("/usr/include/eigen3") 
include_directories ("/usr/include/opencv4/")

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

# Add GIFT
find_package(GIFT QUIET)
if (${GIFT_FOUND})
    message("GIFT is being used as an installed package.")
    set(GIFT_INCLUDE_DIRS "GIFT::GIFT")
else()
    message("GIFT is being used as a git submodule.")
    execute_process(COMMAND git submodule update --init --recursive --remote
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/GIFT
                    RESULT_VARIABLE GIFT_SUBMOD_RESULT)
    if (${GIFT_SUBMOD_RESULT})
        message(FATAL_ERROR "GIFT was not found as a package and could not be included as a submodule.")
    endif()
    add_subdirectory(external/GIFT)
    set(GIFT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/GIFT/GIFT/include")
endif()

# Add LiePP
message("LiePP is being used as a git submodule.")
execute_process(COMMAND git submodule update --init --recursive --remote
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/LiePP
                RESULT_VARIABLE LiePP_SUBMOD_RESULT)
if (${argparse_SUBMOD_RESULT})
    message(FATAL_ERROR "LiePP could not be included as a submodule.")
endif()
set(LiePP_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/external/LiePP/include")

add_subdirectory(libs)

set(AP_VIO_HEADER_FILES
    include/eqvio/csv/CSVLine.h
    include/eqvio/csv/CSVReader.h

    include/eqvio/common/aofstream.h
    include/eqvio/common/LieYaml.h 
    include/eqvio/common/safeConfig.h 

    include/eqvio/mathematical/VIOState.h
    include/eqvio/mathematical/IMUVelocity.h
    include/eqvio/mathematical/VIOGroup.h
    include/eqvio/mathematical/VisionMeasurement.h
    include/eqvio/mathematical/EqFMatrices.h
    include/eqvio/mathematical/Geometry.h
    include/eqvio/mathematical/VIO_eqf.h

    include/eqvio/VIOFilter.h
    include/eqvio/VIOFilterSettings.h
    include/eqvio/dataStream.h
    include/eqvio/VIOWriter.h
    include/eqvio/LoopTimer.h
    )
set(AP_VIO_SOURCE_FILES
    src/mathematical/VIOState.cpp
    src/mathematical/VIOGroup.cpp
    src/mathematical/VisionMeasurement.cpp
    src/mathematical/IMUVelocity.cpp
    src/mathematical/EqFMatrices.cpp
    src/mathematical/Geometry.cpp
    src/mathematical/VIO_eqf.cpp

    src/mathematical/coordinateSuite/euclid.cpp
    src/mathematical/coordinateSuite/invdepth.cpp
    src/mathematical/coordinateSuite/normal.cpp

    src/VIOFilter.cpp
    src/dataStream.cpp
    src/VIOWriter.cpp
    src/LoopTimer.cpp
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
    PUBLIC ${LiePP_INCLUDE_DIRS}
    PUBLIC ${GIFT_INCLUDE_DIRS}
)

target_link_libraries(ap_vio_lib
    ${MODULE_LIBS}
    ${OpenCV_LIBS}
)


find_package(OpenCV REQUIRED)
find_package(GIFT REQUIRED)

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
    PRIVATE ${LiePP_INCLUDE_DIRS}
    PRIVATE ${GIFT_INCLUDE_DIRS}
)

