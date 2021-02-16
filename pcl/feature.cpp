#include <iostream>
#include <sstream>
#include <string>
#include <list>
#include <fstream>
#include <chrono>
#include <vector>

#include "include/io.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/features/eigen.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/features/fpfh.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/search/search.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>


/* ############################################################################
   #####                                                                   ####
   #####                            FEATURES                               ####
   #####                                                                   ####
   ######################################################################### */

void compute_principale_curvature_estimation(std::string input_filename,
                                             std::string output_filename,
                                             float radius)
{
    /* ########################################################
       ##                    READ FILE                      ##
       ######################################################## */

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_file(input_filename);

    /* ########################################################
       ##           COMPUTE  NORMALS                         ##
       ######################################################## */

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normalEstimation.setSearchMethod(tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 10 cm
    normalEstimation.setRadiusSearch(radius);

    // Compute the features
    normalEstimation.compute(*normals);

    /* ########################################################
       ##       COMPUTE  PRINCIPAL CURVATURE ESTIMATION      ##
       ######################################################## */

    // Setup the principal curvatures computation
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principalCurvaturesEstimation;

    // Provide the original point cloud (without normals)
    principalCurvaturesEstimation.setInputCloud(cloud);

    // Provide the point cloud with normals
    principalCurvaturesEstimation.setInputNormals(normals);

    // Use the same KdTree from the normal estimation
    principalCurvaturesEstimation.setSearchMethod(tree);
    principalCurvaturesEstimation.setRadiusSearch(radius);

    // Actually compute the principal curvatures
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());
    principalCurvaturesEstimation.compute(*principalCurvatures);

    /* ########################################################
       ##                    WRITE FILE                      ##
       ######################################################## */

    write_file(input_filename, output_filename, principalCurvatures);
}

void compute_covariance_feature(std::string filename)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_file(filename);

    // Placeholder for the 3x3 covariance matrix at each surface patch
    Eigen::Matrix3f covariance_matrix;
    // 16-bytes aligned placeholder for the XYZ centroid of a surface patch
    Eigen::Vector4f xyz_centroid;

    EIGEN_ALIGN16 Eigen::Vector3f eigen_value;
    EIGEN_ALIGN16 Eigen::Vector3f eigen_vector;

    // Estimate the XYZ centroid
    pcl::compute3DCentroid(*cloud, xyz_centroid);

    // Compute the 3x3 covariance matrix
    pcl::computeCovarianceMatrix(*cloud, xyz_centroid, covariance_matrix);
    // Compute eigen value
    pcl::eigen33(covariance_matrix, eigen_value);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    kdtree.setInputCloud(cloud);

    pcl::PointXYZ searchPoint;

    // Neighbors within radius search

    std::vector<int> nn_indices(10000);
    std::vector<float> nn_dists(10000);

    float radius = 0.10;

    pcl::PointCloud<pcl::PointXYZ> subsetcloud;
    subsetcloud.height = 1;
    subsetcloud.is_dense = false;

    float eigenvalues[cloud->points.size()][3];

    float progress = 0;

    for (std::size_t idx = 0; idx < cloud->points.size(); ++idx)
    {
        if (idx > cloud->points.size() * progress)
        {
            std::cout << progress << std::endl;
            progress += 0.10;
        }

        searchPoint.x = cloud->points[idx].x;
        searchPoint.y = cloud->points[idx].y;
        searchPoint.z = cloud->points[idx].z;

        if (kdtree.radiusSearch(searchPoint, radius, nn_indices, nn_dists) > 0)
        {

            // subsetcloud.width = nn_indices.size();
            // subsetcloud.points.resize (subsetcloud.width);

            // xyz_centroid[0] = 0;
            // xyz_centroid[1] = 0;
            // xyz_centroid[2] = 0;

            // for (int j = 0; j < nn_indices.size(); j++)
            // {
            //     subsetcloud.points[j].x = cloud->points[nn_indices[j]].x;
            //     subsetcloud.points[j].y = cloud->points[nn_indices[j]].y;
            //     subsetcloud.points[j].z = cloud->points[nn_indices[j]].z;

            //     xyz_centroid[0] += subsetcloud.points[j].x;
            //     xyz_centroid[1] += subsetcloud.points[j].y;
            //     xyz_centroid[2] += subsetcloud.points[j].z;
            // }
            // xyz_centroid[0] /= float(nn_indices.size());
            // xyz_centroid[1] /= float(nn_indices.size());
            // xyz_centroid[2] /= float(nn_indices.size());

            // Estimate the XYZ centroid
            pcl::compute3DCentroid(*cloud, nn_indices, xyz_centroid);

            // Compute the 3x3 covariance matrix
            //pcl::computeCovarianceMatrix(*cloud, nn_indices, xyz_centroid, covariance_matrix);
            // Compute eigen value
            // pcl::eigen33 (covariance_matrix, eigen_value);

            // eigenvalues[i][0] = eigen_value[0];
            // eigenvalues[i][1] = eigen_value[1];
            // eigenvalues[i][2] = eigen_value[2];
            continue;
        }
    }
}

void compute_fast_point_feature_histograms(std::string input_filename,
                                           std::string output_filename,
                                           float radius)
{

    /* ########################################################
       ##                    READ FILE                      ##
       ######################################################## */

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_file(input_filename);

    /* ########################################################
       ##                COMPUTE  NORMALS                    ##
       ######################################################## */

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    normalEstimation.setSearchMethod(tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 10 cm
    normalEstimation.setRadiusSearch(radius);

    // Compute the features
    normalEstimation.compute(*normals);

    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);

    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());
    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch(radius);
    // Compute the features
    fpfh.compute(*fpfhs);

    /* ########################################################
       ##                   WRITE IT                         ##
       ######################################################## */

    write_file(input_filename, output_filename, fpfhs);
}

/* ############################################################################
   #####                                                                   ####
   #####                            SEGMENTATION                           ####
   #####                                                                   ####
   ######################################################################### */

void euclidian_clustering(std::string input_filename,
                          std::string output_filename)
{
    /* ########################################################
       ##                    READ FILE                      ##
       ######################################################## */

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_file(input_filename);
    pcl::PCDWriter writer;

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.005); // 5mm
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    int j = 1;
    // CREATE vector of size pointcloude initialize to zeros
    std::cout << cloud->points.size() << std::endl;
    int cluster_number[cloud->points.size()];
    for (int i = 0; i < cloud->points.size(); ++i)
    {
        cluster_number[i] = 0;
    }

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            cluster_number[*pit] = j;
            cloud_cluster->points.push_back(cloud->points[*pit]);
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
        // std::stringstream ss;
        // ss << "cloud_cluster_" << j << ".pcd";
        // writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false);

        j++;
    }

    write_file(input_filename, output_filename, cluster_number);
}

void growing_segmentation(std::string input_filename,
                          std::string output_filename)
{
    /* ########################################################
       ##                    READ FILE                      ##
       ######################################################## */

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_file(input_filename);
    pcl::PCDWriter writer;

    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setRadiusSearch(0.05);
    normal_estimator.compute(*normals);

    // pcl::IndicesPtr indices(new std::vector<int>);
    // pcl::PassThrough<pcl::PointXYZ> pass;
    // pass.setInputCloud(cloud);
    // pass.setFilterFieldName("z");
    // pass.setFilterLimits(0.0, 1.0);
    // pass.filter(*indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(100);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(50);
    reg.setInputCloud(cloud);
    //reg.setIndices (indices);
    reg.setInputNormals(normals);

    reg.setSmoothnessThreshold(10.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;
    std::cout << "These are the indices of the points of the initial" << std::endl
              << "cloud that belong to the first cluster:" << std::endl;
    int j = 1;
    // CREATE vector of size pointcloude initialize to zeros
    std::cout << cloud->points.size() << std::endl;
    int cluster_number[cloud->points.size()];
    for (int i = 0; i < cloud->points.size(); ++i)
    {
        cluster_number[i] = 0;
    }

    for (std::vector<pcl::PointIndices>::const_iterator it = clusters.begin(); it != clusters.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            cluster_number[*pit] = j;
            cloud_cluster->points.push_back(cloud->points[*pit]);
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
        // std::stringstream ss;
        // ss << "cloud_cluster_" << j << ".pcd";
        // writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false);

        j++;
    }

    write_file(input_filename, output_filename, cluster_number);
}

void print_usage()
{
    std::cout << "geometric_feature [feature_name] [radius] [input_filename] [outputfilename]" << std::endl;
    std::cout << "\t[feature_name] : Available feature -> fpfh curvature" << std::endl;
    std::cout << "\f[radius] : Size of the neigborhood search -> Example 0.10 (10cm)" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc == 5)
    {
        float radius = std::stof(argv[2]);
        std::string feature = std::string(argv[1]);

        if (feature == "fpfh")
        {
            compute_fast_point_feature_histograms(argv[3], argv[4], radius);
        }
        if (feature == "curvature")
        {
            compute_principale_curvature_estimation(argv[3], argv[4], radius);
        }

        // euclidian_clustering(argv[1], argv[2]);
        // auto t3 = std::chrono::high_resolution_clock::now();
        // auto duration_2 = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
        // std::cout << "Compute euclidian clustering : " << duration_2 << std::endl;

        // growing_segmentation(argv[1], argv[2]);
        // auto t3 = std::chrono::high_resolution_clock::now();
        // auto duration_2 = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
        // std::cout << "Compute euclidian clustering : " << duration_2 << std::endl;

        /* compute_covariance_feature("../data.txt");
        auto t4 = std::chrono::high_resolution_clock::now();
        auto duration_3 = std::chrono::duration_cast<std::chrono::seconds>( t4 - t3 ).count();
        std::cout << "Compute covariance feature : " << duration_3 << std::endl; */
    }
    else
    {
        print_usage();
    }
}
