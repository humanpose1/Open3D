#include <cmath>

#include "Bound.h"
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Utility/Eigen.h>
#include <Open3D/Geometry/KDTreeFlann.h>
#include <Open3D/Utility/Console.h>

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

	    double error = 0.0;
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
			double gamma_i = 2 * source.points_[i].norm();
			if(0.5 * SQRT3 * size_rotation_cube < PI * 0.5)
			    gamma_i *= std::sin(0.5 * SQRT3 * size_rotation_cube);
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
		    
	    return error/counter;
	}
    } // namespace registration 
} // namespace open3d
