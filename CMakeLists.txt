cmake_minimum_required(VERSION 3.10)
project(PoissonReverb)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER g++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3 -funroll-loops -march=native -mfpmath=sse")

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(LV2_DIR /usr/lib64/lv2)
set(install_dir ${LV2_DIR}/natural_reverb.lv2)

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

add_library(reverb SHARED reverb.cpp reverb.peg)
set(reverb_headers
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	${LVTK_INCLUDE_DIR})
set(reverb_libs
	Threads::Threads
	Eigen3::Eigen
	-lzita-convolver)
target_include_directories(reverb PRIVATE ${reverb_headers})
target_link_libraries(reverb PRIVATE ${reverb_libs})
install(TARGETS reverb
	DESTINATION ${install_dir})
install(FILES
	${CMAKE_CURRENT_SOURCE_DIR}/reverb.ttl
	${CMAKE_CURRENT_SOURCE_DIR}/manifest.ttl
	DESTINATION ${install_dir})

set_source_files_properties(reverb.ttl PROPERTIES LANGUAGE TTL)

add_executable(tests reverb_test.cpp reverb.peg)
target_link_libraries(tests PRIVATE
	${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
	${reverb_libs})
target_include_directories(tests PUBLIC ${reverb_headers})
add_test(NAME test COMMAND tests)
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
enable_testing()
