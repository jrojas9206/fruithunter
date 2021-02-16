#include "include/io.hpp"

/* ############################################################################
   #####                                                                   ####
   #####                            IO                                     ####
   #####                                                                   ####
   ######################################################################### */

pcl::PointCloud<pcl::PointXYZ>::Ptr load_file(std::string filename)
{
    std::ifstream infile(filename);
    std::string line;

    int number_of_line = 0;
    std::list<xyz> l_xyz;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        float x, y, z;
        if (!(iss >> x >> y >> z))
        {
            break;
        } // error
        number_of_line++;
        xyz point = {x, y, z};
        l_xyz.push_back(point);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the cloud data
    cloud->width = number_of_line;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (std::size_t i = 0; i < cloud->points.size(); ++i)
    {
        xyz point = l_xyz.front();
        cloud->points[i].x = point.x;
        cloud->points[i].y = point.y;
        cloud->points[i].z = point.z;
        l_xyz.pop_front();
    }

    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr load_labeled_file(std::string filename)
{
    std::ifstream infile(filename);
    std::string line;

    std::list<xyzl> l_xyzl;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        float x, y, z, label;
        if ((iss >> x >> y >> z >> label))
        {
            xyzl point = {x, y, z, label};
            l_xyzl.push_back(point);
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    int number_of_line = 0;
    std::list<xyzl>::iterator it;
    for (it = l_xyzl.begin(); it != l_xyzl.end(); ++it)
    {
        if (it->label == 1)
        {
            number_of_line++;
        }
    }

    // Fill in the cloud data
    cloud->width = number_of_line;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    int i = 0;
    for (it = l_xyzl.begin(); it != l_xyzl.end(); ++it)
    {
        if (it->label == 1)
        {
            cloud->points[i].x = it->x;
            cloud->points[i].y = it->y;
            cloud->points[i].z = it->z;
            i++;
        }
    }

    return cloud;
}

void write_file(std::string input_file,
                std::string output_file,
                pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures)
{
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);
    std::string line;

    if (infile.is_open() && outfile.is_open())
    {
        int i = 0;
        while (std::getline(infile, line))
        {
            outfile << line << " ";

            outfile << line << " "
                    << principalCurvatures->points[i].principal_curvature_x << " "
                    << principalCurvatures->points[i].principal_curvature_y << " "
                    << principalCurvatures->points[i].principal_curvature_z << " "
                    << principalCurvatures->points[i].pc1 << " "
                    << principalCurvatures->points[i].pc2 << std::endl;
            i++;
        }
        outfile.close();
        infile.close();
    }
}

void write_file(std::string input_file,
                std::string output_file,
                pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs)
{
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);
    std::string line;

    if (infile.is_open() && outfile.is_open())
    {
        int i = 0;
        while (std::getline(infile, line))
        {

            outfile << line << " ";

            for (std::size_t j = 0; j < 33; ++j)
            {
                outfile << fpfhs->points[i].histogram[j] << " ";
            }
            outfile << std::endl;
            i++;
        }
        outfile.close();
        infile.close();
    }
}

void write_file(std::string input_file,
                std::string output_file,
                int *label)
{
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);
    std::string line;

    if (infile.is_open() && outfile.is_open())
    {
        int i = 0;
        while (std::getline(infile, line))
        {

            outfile << line << " ";
            outfile << label[i] << std::endl;
            i++;
        }
        outfile.close();
        infile.close();
    }
}

void write_labeled_file(std::string output_file,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                        int *label)
{
    std::ofstream outfile(output_file);
    std::string line;

    if (outfile.is_open())
    {
        int i = 0;
        for (int i = 0; i < cloud->points.size(); ++i)
        {
            outfile << cloud->points[i].x << " "
                    << cloud->points[i].y << " "
                    << cloud->points[i].z << " "
                    << label[i] << std::endl;
        }
    }
    outfile.close();
}