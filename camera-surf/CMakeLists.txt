cmake_minimum_required(VERSION 2.8)
project( CAMERA_SURF )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable( exec src.cxx )
target_link_libraries( exec ${OpenCV_LIBS} )
