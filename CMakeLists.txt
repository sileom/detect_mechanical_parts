cmake_minimum_required(VERSION 3.0.2)
project(detect_mechanical_parts)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(OpenCV 4.4.0 REQUIRED
   NO_MODULE #Should be optional, tells CMake to use config mode
   PATHS /usr/local/lib/cmake/opencv4 # Tells CMake to look here
   NO_DEFAULT_PATH #and don't look anywhere else
)

#set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
#find_package(OpenCV 4.4.0 REQUIRED)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

add_message_files(
  FILES
  SubscribeToStreamMsg.msg
  AckMsg.msg
  DetectionMsg.msg
  DetectionMsgNoHeader.msg
  DetectionArrayMsg.msg
  ImageReadyMsg.msg
  BoxesMsg.msg
  DisplacementMsg.msg
  hmi/CoordinateMsg.msg
  hmi/ComponenteErrorMsg.msg
  hmi/PrelievoComponenteMsg.msg
  hmi/DepositoComponenteMsg.msg
  hmi/RichiestaCoordinateMsg.msg
  hmi/RichiestaCoordinateStatusMsg.msg
  hmi/ComponenteStatusMsg.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs  
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES detect_mechanical_parts
  CATKIN_DEPENDS message_runtime cv_bridge image_transport rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
