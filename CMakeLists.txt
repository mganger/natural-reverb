cmake_minimum_required(VERSION 3.10)
project(PoissonReverb)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
set(reverb_headers ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${LVTK_INCLUDE_DIR})
target_include_directories(reverb PUBLIC ${reverb_headers})
message(STATUS "${reverb_headers}")

set_source_files_properties(reverb.ttl PROPERTIES LANGUAGE TTL)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
add_executable(tests reverb_test.cpp)
target_link_libraries(tests PRIVATE ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
target_include_directories(tests PUBLIC ${reverb_headers})
add_test(NAME test COMMAND tests)
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
enable_testing()
