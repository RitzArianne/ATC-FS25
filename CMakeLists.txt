cmake_minimum_required(VERSION 3.10)

# Project name and version
project(ATC-FS25 VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the include/ directory
include_directories(include)

# Add the executable
#add_executable(my_cpp_executable src/main.cpp)

file(GLOB SOURCES "src/*.cpp")

add_executable(my_cpp_executable ${SOURCES})