cmake_minimum_required(VERSION 2.8)

project(Watershed)

#################
# Safety Checks #
#################

# Avoid in-source builds
if ("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(FATAL_ERROR "Warning: In-source build! Aborting. Call cmake .. from a build folder.")
endif()

#######################
# CMake path packages #
#######################

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

#####################
# Compiler Settings #
#####################

message("Setting compiler settings:")
if( WIN32 )
    message("-- Windows detected")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    message("-- Linux/UNIX detected")
    # Set C++11
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    # Enable all warnings
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

endif()
message("-- cxx Flags: " ${CMAKE_CXX_FLAGS})
message("-- Finished setting compiler settings.")

###############
# Directories #
###############

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

#########
# Files #
#########
set(SOURCES_CPP 
    ${SOURCE_DIR}/Dummy.cpp
    ${SOURCE_DIR}/lodepng.cpp)
set(INCLUDES_HPP
    ${INCLUDE_DIR}/Dummy.hpp
    ${INCLUDE_DIR}/lodepng.h)

# all files
set(SOURCES_ALL ${SOURCES_CPP} ${INCLUDES_DIR})

add_executable(Watershed main.cpp ${SOURCES_ALL})

set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("-- OpenCV found")
endif()

#find_package(SDL2 REQUIRED)
#add_definitions(-lSDL2main -lSDL2)
add_definitions( -fopenmp )

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_objdetect)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


#FIND_PACKAGE(GDCM REQUIRED)
#   IF(GDCM_FOUND)
#       INCLUDE(${GDCM_USE_FILE})
#       SET(GDCM_LIBRARIES 
#           gdcmcharls
#           gdcmCommon
#           gdcmDICT
#           gdcmDSED
#           gdcmexpat
#           #gdcmgetopt
#           gdcmIOD
#           gdcmjpeg12
#           gdcmjpeg16
#           gdcmjpeg8
#           gdcmMEXD
#           gdcmMSFF
#           gdcmopenjpeg
#           gdcmzlib)
#   ELSE(GDCM_FOUND)
#       MESSAGE(FATAL_ERROR "Cannot find GDCM, did you set GDCM_DIR?")
#   ENDIF(GDCM_FOUND)


#add_definitions(-lgdcmDSED -lgdcmDICT -lgdcmIOD -lgdcmMSFF -lgdcmMEXD)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(Watershed ${OpenCV_LIBS} ${GDCM_LIBRARIES})
