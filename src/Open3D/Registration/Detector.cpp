// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Feature.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/KDTreeFlann.h>
#include <iostream>

namespace open3d {
    namespace {
	using namespace geometry;
	double sqr(double x) { return x * x; }
	
	Eigen::Vector3d
	FastEigen3x3(const Eigen::Matrix3d &A) {
	    // Based on:
	    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
	    // the problem is when two eigenvalue or more are equal. It can be a bug
	    double p1 = sqr(A(0, 1)) + sqr(A(0, 2)) + sqr(A(1, 2));
	    Eigen::Vector3d eigenvalues;
	    if (p1 == 0.0) {
		eigenvalues(2) = std::min(A(0, 0), std::min(A(1, 1), A(2, 2)));
		eigenvalues(0) = std::max(A(0, 0), std::max(A(1, 1), A(2, 2)));
		eigenvalues(1) = A.trace() - eigenvalues(0) - eigenvalues(2);
	    } else {
		double q = A.trace() / 3.0;
		double p2 = sqr((A(0, 0) - q)) + sqr(A(1, 1) - q) + sqr(A(2, 2) - q) +
                    2 * p1;
		double p = sqrt(p2 / 6.0);
		Eigen::Matrix3d B = (1.0 / p) * (A - q * Eigen::Matrix3d::Identity());
		double r = B.determinant() / 2.0;
		double phi;
		if (r <= -1) {
		    phi = M_PI / 3.0;
		} else if (r >= 1) {
		    phi = 0.0;
		} else {
		    phi = std::acos(r) / 3.0;
		}
		eigenvalues(0) = q + 2.0 * p * std::cos(phi);
		eigenvalues(2) = q + 2.0 * p * std::cos(phi + 2.0 * M_PI / 3.0);
		eigenvalues(1) = q * 3.0 - eigenvalues(0) - eigenvalues(2);
	    }
	    
	    return eigenvalues;
	}
	Eigen::Matrix3d ComputeCovariance(const PointCloud &cloud,
				      const std::vector<int> &indices) {
	    if (indices.size() == 0) {
		return Eigen::Matrix3d::Zero();
	    }
	    Eigen::Matrix3d covariance;
	    Eigen::Matrix<double, 9, 1> cumulants;
	    cumulants.setZero();
	    for (size_t i = 0; i < indices.size(); i++) {
		const Eigen::Vector3d &point = cloud.points_[indices[i]];
		cumulants(0) += point(0);
		cumulants(1) += point(1);
		cumulants(2) += point(2);
		cumulants(3) += point(0) * point(0);
		cumulants(4) += point(0) * point(1);
		cumulants(5) += point(0) * point(2);
		cumulants(6) += point(1) * point(1);
		cumulants(7) += point(1) * point(2);
		cumulants(8) += point(2) * point(2);
	    }
	    cumulants /= (double)indices.size();
	    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
	    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
	    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
	    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
	    covariance(1, 0) = covariance(0, 1);
	    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
	    covariance(2, 0) = covariance(0, 2);
	    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
	    covariance(2, 1) = covariance(1, 2);

	    return covariance;
    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    // solver.compute(covariance, Eigen::ComputeEigenvectors);
    // return solver.eigenvectors().col(0);
}
	void NMS(std::shared_ptr<registration::Feature> &saliency,
		 const std::vector<int> &indices_max,
		 const geometry::PointCloud &input,
		 const geometry::KDTreeFlann &kdtree,
		 const geometry::KDTreeSearchParam &search_param) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	    for(int i = 0; i < indices_max.size(); i++){
		std::vector<int> indices;
		std::vector<double> distance2;
		const auto &point = input.points_[indices_max[i]];
		kdtree.Search(point, search_param, indices, distance2);
		
		for(int j = 0; j < indices.size(); j++) {
		    if(saliency->data_(0, indices[j])
		       > saliency->data_(0, indices_max[i])) {
			saliency->data_(0, indices_max[i]) = 0.0;
			 // one point has a bigger saliency
			continue;
		    }
		}
		
	    }
	    
	}
	
    } // unnamed namespace 
    namespace registration {
	
	std::shared_ptr<Feature> SaliencyByCovariance(
	    const geometry::PointCloud &input,
	    const geometry::KDTreeSearchParam &search_param_cov,
	    const geometry::KDTreeSearchParam &search_param_NMS,
	    double gamma_01, double gamma_12) {
	    auto saliency = std::make_shared<Feature>();
	    saliency->Resize(1, (int)input.points_.size());
	    geometry::KDTreeFlann kdtree(input);
	    std::vector<int> indices_max_saliency;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	    for(int i = 0; i < (int)input.points_.size(); i++) {
		
		const auto &point = input.points_[i];
		std::vector<int> indices;
		std::vector<double> distance2;
		kdtree.Search(point, search_param_cov, indices, distance2);
		Eigen::Matrix3d covariance = ComputeCovariance(input, indices);
		Eigen::Vector3d eigenvalues = FastEigen3x3(covariance);
		
		if(eigenvalues(1)/eigenvalues(0) < gamma_01 and
		   eigenvalues(2)/eigenvalues(1) < gamma_12) {
		    saliency->data_(0, i) = eigenvalues(2);
		    
		}
		else {
		    saliency->data_(0, i) = 0.0;
		}
		
	    }
	    
	    for(int i = 0; i < (int)input.points_.size(); i++){
		if(saliency->data_(0, i) > 0.0)
		    indices_max_saliency.push_back(i);
	    }
	    
	    std::cout << "compute saliency of" << std::endl;
	    NMS(saliency, indices_max_saliency,
	    input, kdtree, search_param_NMS);
	    std::cout << "NMS" << std::endl;
	    return saliency;
	}
    }// namespace registration
} // namespace open3d
