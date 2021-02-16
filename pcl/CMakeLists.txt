cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project("geometric_feature")

#set(PCL_DIR "/usr/local/share/pcl-1.10/")

# set(CONDA_PREFIX "/gpfswork/rech/wwk/uqr22pt/.conda/envs/fruithunter") 

set(PCL_INCLUDE_DIRS ${CONDA_PREFIX}."/include/pcl-1.9.1/")
set(PCL_LIBRARY_DIRS ${CONDA_PREFIX}."/lib/")


# set(EIGEN_INCLUDE_DIR "/gpfswork/rech/wwk/uqr22pt/fruithunter/eigen/build/eigen3/include/eigen3/")
# set(EIGEN_LIBRARY_DIRS ${CONDA_PREFIX}."/lib/")
# set(EIGEN_ROOT "/gpfswork/rech/wwk/uqr22pt/fruithunter/eigen/build/eigen3")
# set(EIGEN_DIR "/gpfswork/rech/wwk/uqr22pt/fruithunter/eigen/build/eigen3/share/eigen3/cmake/")

set (CMAKE_CXX_STANDARD 11)

#set_property(TARGET tgt PROPERTY CXX_STANDARD 11)

find_package(PCL 1.9.1 REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
# find_package(VTK COMPONENTS system REQUIRED)
# find_package(EIGEN3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS}  ${MY_LIBRARY_DIRS} ${EIGEN_INCLUDE_DIR})

link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARIES})

add_definitions(${PCL_DEFINITIONS})

add_executable(my_feature feature.cpp io.cpp)
target_link_libraries(my_feature ${PCL_LIBRARIES})

add_executable(my_segmentation segmentation.cpp io.cpp)
target_link_libraries(my_segmentation ${PCL_LIBRARIES})