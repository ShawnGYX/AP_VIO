cmake_minimum_required(VERSION 3.0)
project(AP_VIO VERSION 0.0.1)
set(CMAKE_CXX_STANDARD 17)

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

option( SUPPORT_CONCEPTS "Build EqVIO using c++20 concepts" OFF)
add_compile_definitions(SUPPORT_CONCEPTS=$<BOOL:${SUPPORT_CONCEPTS}>)

find_package(GIFT QUIET)
if (${GIFT_FOUND})
    message("GIFT is being used as an installed package.")
    set(GIFT_INCLUDE_DIRS "GIFT::GIFT")
else()
    message("GIFT is being used as a git submodule.")
    execute_process(COMMAND git submodule update --init --recursive --remote
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/External/GIFT
                    RESULT_VARIABLE GIFT_SUBMOD_RESULT)
    if (${GIFT_SUBMOD_RESULT})
        message(FATAL_ERROR "GIFT was not found as a package and could not be included as a submodule.")
    endif()
    add_subdirectory(External/GIFT)
    set(GIFT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/External/GIFT/GIFT/include")
endif()

# Add LiePP

message("LiePP is being used as a git submodule.")
execute_process(COMMAND git submodule update --init --recursive --remote
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/External/LiePP
                RESULT_VARIABLE LiePP_SUBMOD_RESULT)
if (${argparse_SUBMOD_RESULT})
    message(FATAL_ERROR "LiePP could not be included as a submodule.")
endif()
set(LiePP_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/External/LiePP/include")



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
    include/eqf_vio/Geometry.h
    include/eqf_vio/dataStream.h
    )
set(AP_VIO_SOURCE_FILES
    src/VIOState.cpp
    src/VIOGroup.cpp
    src/VIOFilter.cpp
    src/VisionMeasurement.cpp
    src/IMUVelocity.cpp
    src/EqFMatrices.cpp
    src/Geometry.cpp
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
    PUBLIC ${OPENCV_INCLUDE_DIRS}
    PUBLIC ${GIFT_INCLUDE_DIRS}
    PUBLIC ${LiePP_INCLUDE_DIRS}
    PUBLIC ${EIGEN_INCLUDE_DIRS}
)

target_link_libraries(ap_vio_lib
    ${MODULE_LIBS}
    ${OpenCV_LIBS}
    GIFT
)

if (${CMAKE_COMPILER_IS_GNUCC} AND ${CMAKE_CXX_COMPILER_VERSION} LESS 9.0)
    message("GCC version is less than 9. Manually linking std::filesystem.")
    target_link_libraries(ap_vio_lib stdc++fs)
endif()

# find_package(OpenCV REQUIRED)
# find_package(GIFT REQUIRED)

add_executable(vio_ap src/main_ap.cpp)

target_link_libraries(vio_ap
    GIFT
    ${OpenCV_LIBS}
    ap_vio_lib
    ${MODULE_LIBS}
    yaml-cpp
)
target_include_directories(vio_ap
    PRIVATE include
    PRIVATE ${EIGEN_INCLUDE_DIRS}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE ${YAML_CPP_INCLUDE_DIR}
    PRIVATE ${LiePP_INCLUDE_DIRS}
    PRIVATE ${GIFT_INCLUDE_DIRS}
    PRIVATE ${argparse_INCLUDE_DIRS}
)

