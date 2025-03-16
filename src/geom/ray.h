#ifndef RAY_H
#define RAY_H
#include "vec3.h"

class ray {
    point3 o;
    vector3 d;

public:
    __device__ __host__ ray(const point3 &origin, const vector3 &direction) : o(origin), d(direction) {}

    __device__ __host__ const point3 &origin() const { return o; }
    __device__ __host__ const vector3 &direction() const { return d; }

    __device__ __host__ point3 at(float const t) const { return o + d * t; }
};


#endif