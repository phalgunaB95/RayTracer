#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "../util.h"
#include "hittable.h"

class hittable_list : public hittable {
    hittable **obj_list;//  pointer to array of pointers
    int obj_count;

public:
    __device__ hittable_list(int obj_count) : obj_count(obj_count) {
        obj_list = (hittable **) malloc(obj_count * sizeof(hittable *));
        // checkCudaErrors(cudaMallocManaged((void **) &obj_list, ));
    }

    __device__ hittable_list(hittable **obj_list, int obj_count) : obj_count(obj_count) {
        obj_list       = (hittable **) malloc(obj_count * sizeof(hittable *));
        this->obj_list = new hittable *[obj_count];
        for (int i = 0; i < obj_count; i++) this->obj_list[i] = obj_list[i];
    }

    __device__ bool hit(ray const &R, float tmin, float tmax, hit_record &record) const override {
        auto hit   = false;
        auto hit_t = tmax;
        for (int i = 0; i < obj_count; i++) {
            auto obj = obj_list[i];
            auto tmp = record;
            if (obj->hit(R, tmin, hit_t, tmp)) {
                hit    = true;
                hit_t  = tmp.t;
                record = tmp;
            }
        }
        return hit;
    }
    __device__ int objects_count() const { return obj_count; }
    __device__ hittable **objects() const { return obj_list; }

    __device__ ~hittable_list() {
        free(obj_list);
    }
};

using world = hittable_list;


#endif