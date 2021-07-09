cmake_minimum_required(VERSION 3.0)
project(AP_eqvio VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)

# Set off if you want to build the project in pure cmake


include(CMakeDependentOption)


include_directories(
  include
)


## Declare a C++ library
add_library(uartlib
  mavhelper.h
  mavhelper.cpp
)


add_executable(uart_test main.cpp)
target_link_libraries(uart_test
uartlib
)


# Add the tests
