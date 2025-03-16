#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "../util.h"
#include "hittable.h"

class hittable_list final : public hittable {
    hittable **obj_list;//  pointer to array of pointers
    int obj_count;

public:
    __host__ __device__ explicit hittable_list(int const obj_count) : obj_count(obj_count) {
        obj_list = static_cast<hittable **>(malloc(obj_count * sizeof(hittable *)));
        // checkCudaErrors(cudaMallocManaged((void **) &obj_list, ));
    }

    __host__ __device__ hittable_list(hittable **obj_list, int const obj_count) : obj_count(obj_count) {
        this->obj_list = new hittable *[obj_count];
        for (int i = 0; i < obj_count; i++) this->obj_list[i] = obj_list[i];
    }

    __host__ __device__ bool hit(ray const &R, float const tmin, float const tmax, hit_record &record) const override {
        auto hit   = false;
        auto hit_t = tmax;
        for (int i = 0; i < obj_count; i++) {
            auto const obj = obj_list[i];
            if (auto tmp = record; obj->hit(R, tmin, hit_t, tmp)) {
                hit    = true;
                hit_t  = tmp.t;
                record = tmp;
            }
        }
        return hit;
    }
    __host__ __device__ int objects_count() const { return obj_count; }
    __host__ __device__ hittable **objects() const { return obj_list; }

    __host__ __device__ ~hittable_list() override {
        free(obj_list);
    }
};

using world = hittable_list;


#endif