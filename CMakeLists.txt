cmake_minimum_required(VERSION 3.10)

project(Poisson Reverb)
add_library(reverb reverb.cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
