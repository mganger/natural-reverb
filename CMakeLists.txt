cmake_minimum_required(VERSION 3.10)

find_path(
	LVTK_INCLUDE_DIR
	PATHS /usr/local/include/lvtk-2
	NAMES "lvtk/plugin.hpp"
)
message(STATUS "${LVTK_INCLUDE_DIR}")

project(PoissonReverb)
add_library(reverb OBJECT reverb.cpp)
target_include_directories(reverb PUBLIC
	${LVTK_INCLUDE_DIR})

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
