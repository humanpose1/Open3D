#include "PointCloud.h"
#include <unordered_map>
#include <random>
#include <cmath>

namespace open3d {
namespace geometry {
    std::vector<int> NormalSample(const PointCloud &input,
				  int num_pt,
				  double max_angle,
				  int num_part){
	
	auto normals = input.normals_;
	std::vector<int> output(num_pt);
	// for(int i=0; i<num_part+1; i++){
	//     double u = max_angle * (double) i /num_part;
	//     list_theta.push_back(acos(1-2*u));
	// }
	std::unordered_map<int, std::vector<int> > map_point_section;
	std::vector<int> num_point_section(num_part, 0);

	// init the map
	for (int i=0; i<num_part; i++){
	    std::vector<int> empty;
	    map_point_section[i] = empty;
	}

	// put the points on the right "box"
	for (int i = 0; i < (int)input.points_.size(); i++) {
	    auto pt = input.points_[i];
	    double elevation = atan2(sqrt(pt[0]*pt[0] + pt[1]*pt[1]), pt[2]);
	    int num_box = (int)(0.5*(1-cos(elevation))*num_part/max_angle);
	    if(num_box<num_part){
		map_point_section[num_box].push_back(i);
		num_point_section[num_box] += 1;
	    }
	}
	std::vector<int> not_null_num_pt_section;
	for (int i=0; i<num_part; i++){
	    if(num_point_section[i]>0){
		not_null_num_pt_section.push_back(i);
	    }
	}	
	std::default_random_engine generator1;
	std::uniform_int_distribution<int> distribution1(0,
							(int)not_null_num_pt_section.size());
	for (int i=0; i<num_pt; i++){
	    int num = distribution1(generator1);
	    std::vector<int> list_ind = map_point_section[num];
	    std::default_random_engine generator2;
	    std::uniform_int_distribution<int> distribution2(0,
							     (int)list_ind.size());
	    int num2 = distribution2(generator2);
	    output[i] = list_ind[num2];
	}
	return output;
    }

}// end of geometry
} // end of open3d
