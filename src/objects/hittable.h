#ifndef HITTABLE_H
#define HITTABLE_H

#include "../geom/ray.h"
#include "../geom/vector.h"

struct hit_record {
    point3 P;
    vector3 N;
    float t;
    bool front_face;


    __device__ void set_face_normal(ray const &R, vector3 const &normal) {
        front_face = R.direction().dot(normal) < 0;//  front face facing camera, ray going out the camera, so opposite directions

        N = normal;
        if (!front_face) N *= -1;
    }
};

class hittable {
public:
    // __device__ virtual ~hittable()                                                              = default;
    __device__ virtual bool hit(ray const &r, float tmin, float tmax, hit_record &record) const = 0;
};

#endif