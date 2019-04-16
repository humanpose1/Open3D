#pragma once

#include <vector>
#include <tuple>
#include <Eigen/Core>

#include <Open3D/Geometry/KDTreeFlann.h>
#include <Open3D/Geometry/PointCloud.h>

namespace open3d {

    namespace registration {

	// Class that compute the upper bound and lower bound of an error
	// according to GO ICP : https://arxiv.org/pdf/1605.03344.pdf
	//Attribute :
	// target_: the point cloud
	// is_point2point: integer that indicate if the loss is
	// classical loss or it is point to plane loss

	// methods
	//ComputeBound compute the lower bound and upperbound 
	// wrt to papameters
	
	class ErrorMeasurer{
	public:
	    ErrorMeasurer(const geometry::PointCloud target,
			  int is_point2point);
	    
		
	    ~ErrorMeasurer();
	public:
		double ComputeBound(const geometry::PointCloud &source,
				    double size_rotation_cube,
				    double size_translation_cube);
		    
		
	public:
		geometry::PointCloud target_;
		int is_point2point_;
		geometry::KDTreeFlann tree_;
		
		
	};

    } // namespace registation
} // namespace open3d
