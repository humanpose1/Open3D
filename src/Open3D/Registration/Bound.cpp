#include <cmath>

#include "Bound.h"
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Utility/Eigen.h>
#include <Open3D/Geometry/KDTreeFlann.h>
#include <Open3D/Utility/Console.h>
#include <iostream>

#define SQRT3 1.73205080757
#define PI 3.14159265359
namespace open3d{
    namespace registration{
	ErrorMeasurer::ErrorMeasurer(const geometry::PointCloud target,
				     int is_point2point){
	    
	    target_ = target;
	    is_point2point_ = is_point2point;
	    tree_.SetGeometry(target_);

	}
	ErrorMeasurer::~ErrorMeasurer() {}
	double ErrorMeasurer::ComputeBound(const geometry::PointCloud &source,
					   double size_rotation_cube,
					   double size_translation_cube) {

	    long double error = 0.0;
	    double counter = 0.0;
	    if(!target_.HasNormals() && is_point2point_ == 0){
		utility::PrintDebug(
                "[ErrorMeasurer::ComputeBound] You need normals.\n");
		return -1.0;
	    }
#ifdef _OPENMP
#pragma omp parallel
	    {
#endif
        
#ifdef _OPENMP
#pragma omp for nowait
#endif
		for (int i = 0; i < (int)source.points_.size(); i++) {
		    
		    std::vector<int> indices(1);
		    std::vector<double> dists(1);
		    const auto &point = source.points_[i];
		    if (tree_.SearchKNN(point,
					1, indices, dists) > 0) {
			Eigen::Vector3d st = source.points_[i]-target_.points_[indices[0]];
			double ei = 0.0;
			double gamma_i = 0;
			double angle = 0.5 * SQRT3 * size_rotation_cube;
			if( angle > PI * 0.5){
			    angle = PI * 0.5;
			}
			gamma_i = 2 * std::sin(angle) * source.points_[i].norm();
			double gamma_t = SQRT3 * size_translation_cube;
			if(is_point2point_)
			    ei = st.norm();
			else {
			    
			    Eigen::Vector3d normals = target_.normals_[indices[0]];
			    ei = std::abs(normals.dot(st)); // point 2 plane
			}
			ei = ei - gamma_i - gamma_t;
			if (ei > 0) {
			    counter++;
			    error += ei*ei;
			}
		    }
		}
	    }
	    if(counter>0)
		return error/counter;
	    else
		return 0.0;
	}
	std::shared_ptr<Feature> ErrorMeasurer::DistanceField(const geometry::PointCloud &source){
	    auto feature = std::make_shared<Feature>();
	    feature->Resize(1, (int)source.points_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	    for (int i = 0; i < (int)source.points_.size(); i++) {
		std::vector<int> indices(1);
		std::vector<double> dists(1);
		const auto &point = source.points_[i];
		if (tree_.SearchKNN(point,
					1, indices, dists) > 0) {
			Eigen::Vector3d st = source.points_[i]-target_.points_[indices[0]];
			feature->data_(0, i) = st.norm();
		}
	    }
	    return feature;
	}
    } // namespace registration 
} // namespace open3d
