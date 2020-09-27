cmake_minimum_required(VERSION 3.10)
project(PoissonReverb)

find_path(
	LVTK_INCLUDE_DIR
	PATHS /usr/local/include/lvtk-2
	NAMES "lvtk/plugin.hpp"
)
message(STATUS "${LVTK_INCLUDE_DIR}")

add_custom_command(
	OUTPUT reverb.peg
	COMMAND lv2peg ${CMAKE_CURRENT_SOURCE_DIR}/reverb.ttl ${CMAKE_CURRENT_BINARY_DIR}/reverb.peg
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/reverb.ttl
	BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/reverb.peg)

add_library(reverb OBJECT reverb.cpp reverb.peg)
target_link_libraries(reverb reverb_headers)
target_include_directories(
	reverb PUBLIC
	${CMAKE_CURRENT_BINARY_DIR}
	${LVTK_INCLUDE_DIR})

set_source_files_properties(reverb.ttl PROPERTIES LANGUAGE TTL)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
