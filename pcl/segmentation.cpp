#include <string>
#include <list>
#include <vector>
#include <math.h>

#include "include/io.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/features/normal_3d.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

/* ############################################################################
   #####                                                                   ####
   #####                            SEGMENTATION                           ####
   #####                                                                   ####
   ######################################################################### */

float mean_distance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree)
{
    std::vector<int> k_indices;
    std::vector<float> k_sqr_distances;

    float mean_distance = 0;
    for (int i = 0; i < cloud->points.size(); ++i)
    {
        tree->nearestKSearch(i, 2, k_indices, k_sqr_distances);
        mean_distance += k_sqr_distances[1];
    }
    mean_distance /= cloud->points.size();

    return sqrt(mean_distance);
}

void euclidian_clustering(std::string input_filename,
                          std::string output_filename)
{
    /* ########################################################
       ##                    READ FILE                      ##
       ######################################################## */

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_labeled_file(input_filename);
    pcl::PCDWriter writer;

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    float md = mean_distance(cloud, tree);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    // ec.setClusterTolerance(0.015); // 5mm
    // ec.setMinClusterSize(75);

    // PN PARATEMETERS
    float cluster_tolerance = 1.4 * md;
    float cluster_min_size = 75 / (md * 100);

    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(cluster_min_size);

    std::cout << "Cluster Tolerance : " << cluster_tolerance << std::endl;
    std::cout << "Cluster min size : " << cluster_min_size << std::endl;

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
        }
        j++;
    }

    write_labeled_file(output_filename, cloud, cluster_number);
}

void growing_segmentation(std::string input_filename,
                          std::string output_filename)
{
    /* ########################################################
       ##                    READ FILE                      ##
       ######################################################## */

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_labeled_file(input_filename);
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

    // write_file(input_filename, output_filename, cluster_number);
}

bool customRegionGrowing(const pcl::PointNormal &point_a,
                         const pcl::PointNormal &point_b,
                         float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap();
    Eigen::Map<const Eigen::Vector3f> point_b_normal = point_b.getNormalVector3fMap();

    if (std::abs(point_a_normal.dot(point_b_normal)) < 0.50) // RF PARAMETERS
        return (true);

    return false;
}

void conditional_euclidian_clustering(std::string input_filename,
                                      std::string output_filename)
{
    /* ########################################################
       ##                    READ FILE                      ##
       ######################################################## */

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_labeled_file(input_filename);
    pcl::PCDWriter writer;

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    float md = mean_distance(cloud, tree);

    /* ########################################################
       ##           COMPUTE  NORMALS                         ##
       ######################################################## */

    // Create the normal estimation class, and pass the input dataset to it
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud(*cloud, *cloud_with_normals);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setSearchMethod(tree);
    normalEstimation.setRadiusSearch(1.4 * md);
    normalEstimation.compute(*cloud_with_normals);

    /* ########################################################
       ##           SEGMENTATION                             ##
       ######################################################## */

    std::vector<pcl::PointIndices>
        cluster_indices;

    // Set up a Conditional Euclidean Clustering class
    pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec(true);
    cec.setInputCloud(cloud_with_normals);
    cec.setConditionFunction(&customRegionGrowing);

    // RF PARATEMETERS
    // float cluster_tolerance = 1.4 * md;
    // float cluster_min_size = 75 / (md * 100);

    // PN PARATEMETERS
    float cluster_tolerance = 1.4 * md;
    float cluster_min_size = 75 / (md * 100);

    cec.setClusterTolerance(cluster_tolerance);
    cec.setMinClusterSize(cluster_min_size);

    std::cout << "Cluster Tolerance : " << cluster_tolerance << std::endl;
    std::cout << "Cluster min size : " << cluster_min_size << std::endl;

    cec.setMaxClusterSize(10000);
    cec.segment(cluster_indices);

    int j = 1;
    // CREATE vector of size pointcloude initialize to zeros
    std::cout << cloud->points.size() << std::endl;
    int cluster_number[cloud->points.size()];
    for (int i = 0; i < cloud->points.size(); ++i)
    {
        cluster_number[i] = 0;
    }

    std::cout << "Number of cluster: " << cluster_indices.size() << std::endl;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            // cloud_cluster->points.push_back(cloud->points[*pit]);
            cluster_number[*pit] = j;
        }
        j++;
        // cloud_cluster->width = cloud_cluster->points.size();
        // cloud_cluster->height = 1;
        // cloud_cluster->is_dense = false;

        // std::vector<int> inliers;
        // pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(
        //     new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud_cluster));
        // Eigen::VectorXf model_coefficients;

        // pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_s);
        // ransac.setDistanceThreshold(0.03);
        // ransac.setMaxIterations(1000);
        // bool found = ransac.computeModel();

        // if (found)
        // {
        //     ransac.getModelCoefficients(model_coefficients);
        //     ransac.getInliers(inliers);
        //     float ratio = float(inliers.size()) / float(cloud_cluster->points.size());

        //     if (model_coefficients[3] < 100)
        //     {
        //         for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        //         {
        //             cluster_number[*pit] = j;
        //         }
        //         j++;
        //     }
        //     std::cout << "Radius " << model_coefficients[3] << std::endl
        //               << "Radio " << ratio << std::endl;
        // }
    }

    write_labeled_file(output_filename, cloud, cluster_number);
}

void print_usage()
{
    std::cout << "segmentqtion [method_name] [radius] [input_filename] [outputfilename]" << std::endl;
    std::cout << "\t[method_name] : Available feature -> fpfh curvature" << std::endl;
    std::cout << "\f[radius] : Size of the neigborhood search -> Example 0.10 (10cm)" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc == 4)
    {
        std::string method = std::string(argv[1]);
        std::string input_filename = std::string(argv[2]);
        std::string output_filename = std::string(argv[3]);

        if (method == "euclidian_clustering")
        {
            euclidian_clustering(input_filename, output_filename);
        }
        if (method == "conditional_euclidian_clustering")
        {
            conditional_euclidian_clustering(input_filename, output_filename);
        }
        if (method == "growing_segmentation")
        {
            growing_segmentation(input_filename, output_filename);
        }
    }
    else
    {
        print_usage();
    }
}
