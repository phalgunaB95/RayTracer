#ifndef SPHERE_H
#define SPHERE_H
#include "hittable.h"
#include <cmath>

class sphere : public hittable {
    point3 C;
    float r;

public:
    __device__ sphere(point3 const &center, float radius) : C(center), r(radius) {
        if (r < .0f) r = .0f;
    }

    __device__ bool hit(ray const &R, float tmin, float tmax, hit_record &record) const override {

        auto oc = C - R.origin();

        auto a = R.direction().dot(R.direction());
        auto b = -2 * R.direction().dot(oc);
        auto c = oc.dot(oc) - r * r;

        auto discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return false;
        auto sqrtd = std::sqrt(discriminant);


        auto t = (-b - sqrtd) / (2.0f * a);
        if (t < tmin || t > tmax) {
            t = (-b + sqrtd) / (2.0f * a);
            if (t < tmin || t > tmax) return false;
        }

        record.t    = t;
        record.P    = R.at(t);
        auto normal = (record.P - C) / r;
        record.set_face_normal(R, normal);

        return true;
    }
};

#endif