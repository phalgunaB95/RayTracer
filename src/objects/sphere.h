#ifndef SPHERE_H
#define SPHERE_H
#include "hittable.h"
#include <cmath>

class sphere final : public hittable {
    point3 C;
    float r;

public:
    __host__ __device__ ~sphere() override{};

    __host__ __device__ sphere(point3 const &center, float const radius) : C(center), r(radius) {
        if (r < .0f) r = .0f;
    }

    __host__ __device__ bool hit(ray const &R, float const tmin, float const tmax, hit_record &record) const override {

        auto const oc = C - R.origin();

        auto const a = R.direction().dot(R.direction());
        auto const b = -2 * R.direction().dot(oc);
        auto const c = oc.dot(oc) - r * r;

        auto const discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return false;
        auto const sqrtd = std::sqrt(discriminant);


        auto t = (-b - sqrtd) / (2.0f * a);
        if (t < tmin || t > tmax) {
            t = (-b + sqrtd) / (2.0f * a);
            if (t < tmin || t > tmax) return false;
        }

        record.t = t;
        record.P = R.at(t);

        auto const normal = (record.P - C) / r;
        record.set_face_normal(R, normal);

        return true;
    }
};

#endif