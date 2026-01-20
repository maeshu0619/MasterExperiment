#include <iostream>
#include <thread>
#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
using PointXYZRGB = pcl::PointXYZRGB;
using CloudXYZRGB = pcl::PointCloud<PointXYZRGB>;
using PCLVisualizer = pcl::visualization::PCLVisualizer;

int main() {
	
	CloudXYZRGB::Ptr cloud(new CloudXYZRGB);
	createRandomPCD(cloud);

	const double leaf_size = 3.0;
	CloudXYZRGB::Ptr filtered_cloud(new CloudXYZRGB);

	pcl::VoxelGrid<PointXYZRGB>::Ptr grid(new pcl::VoxelGrid<PointXYZRGB>);
	grid->setInputCloud(cloud);
	grid->setLeafSize(leaf_size, leaf_size, leaf_size);
	grid->setSaveLeafLayout(true);
	grid->filter(*filtered_cloud);

	std::cout << "before : " << cloud->width << std::endl;
	std::cout << "after  : " << filtered_cloud->width << std::endl;

	const double shift_x = 6.0;
	for (auto& point : *cloud) {
		point.x -= shift_x;
		point.r = 0;
	}
	for (auto& point : *filtered_cloud) {
		point.x += shift_x;
		point.b = 0;
	}

	PCLVisualizer::Ptr vis(new PCLVisualizer);
	vis->addPointCloud(cloud, "1");
	vis->addPointCloud(filtered_cloud, "2");
	visualize(vis);

	return 0;
}