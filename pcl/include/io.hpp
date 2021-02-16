#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

/* ############################################################################
   #####                                                                   ####
   #####                            STRUCT                                 ####
   #####                                                                   ####
   ######################################################################### */

struct xyz
{
    float x;
    float y;
    float z;
};

struct xyzl
{
    float x;
    float y;
    float z;
    float label;
};

struct xyzs
{
    float x;
    float y;
    float z;
    float s;
};

/* ############################################################################
   #####                                                                   ####
   #####                            IO                                     ####
   #####                                                                   ####
   ######################################################################### */

pcl::PointCloud<pcl::PointXYZ>::Ptr load_file(std::string filename);
pcl::PointCloud<pcl::PointXYZ>::Ptr load_labeled_file(std::string filename);

void write_file(std::string input_file,
                std::string output_file,
                pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures);

void write_file(std::string input_file,
                std::string output_file,
                pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs);

void write_file(std::string input_file,
                std::string output_file,
                int *label);

void write_labeled_file(std::string output_file,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                        int *label);
