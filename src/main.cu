#include "geom/ray.h"
#include "geom/vector.h"

#include "objects/hittable.h"
#include "objects/hittable_list.h"
#include "objects/sphere.h"

#include "image_buffer.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

constexpr auto aspect_ratio = 16.0 / 9.0;

constexpr auto num_pixels_x = 1920;
constexpr auto num_pixels_y = static_cast<int>(num_pixels_x / aspect_ratio);

const point3 origin(0, 0, 0);

constexpr auto v_h = 2.0f;
const vector3 vertical(0, -v_h, 0);

constexpr float v_w = v_h * (aspect_ratio);
const vector3 horizontal(v_w, 0, 0);

constexpr float focal_length = 1.0;

const auto camera_center = origin;
const auto Q             = camera_center - vector3(0, 0, focal_length) - vertical / 2 - horizontal / 2;
const auto d_vertical    = vertical / num_pixels_y;
const auto d_horizontal  = horizontal / num_pixels_x;
const auto pixel00_loc   = Q + d_horizontal / 2 + d_vertical / 2;

// const int max_recursion_depth = 32;
constexpr float inf = std::numeric_limits<float>::max();

__device__ color3 ray_color(ray const &R, world **d_world) {
    hit_record record;
    if ((*d_world)->hit(R, 0, inf, record))
        return .5f * (record.N + color3(1, 1, 1));

    auto unit = R.direction().unit();
    auto a    = 0.5f * (unit.y() + 1);

    return color3(.5, .7, 1) * a + color3(1, 1, 1) * (1 - a);
}

__global__ void render_on_device(color3 *buffer,
                                 world **d_world,
                                 point3 const pixel00_loc, vector3 const dh, vector3 const dv, point3 const camera_center,
                                 int max_x, int max_y) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    auto pixel = pixel00_loc + dv * j + dh * i;
    ray R(camera_center, pixel - camera_center);
    int pixel_index     = j * max_x + i;
    buffer[pixel_index] = ray_color(R, d_world);
}


__global__ void generate_world(world **d_world) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {// code runs only once!
        *d_world        = new hittable_list(2);
        auto d_obj_list = (*d_world)->objects();
        d_obj_list[0]   = new sphere(point3(0, 0, -1), .5f);
        d_obj_list[1]   = new sphere(point3(0, -100.5, -1), 100);
    }
}
__global__ void destroy_world(world **d_world) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {// code runs only once!
        auto obj_count = (*d_world)->objects_count();
        auto obj_list  = (*d_world)->objects();

        for (int i = 0; i < obj_count; i++) delete obj_list[i];
        delete *d_world;
    }
}


int main(int argc, const char **argv) {
    auto img = ImgBuffer(num_pixels_x, num_pixels_y);

    clock_t start = clock();

    std::clog << "RayTracing::Init" << std::endl;
    dim3 threads(8, 8);
    dim3 blocks(1 + (num_pixels_x / threads.x), 1 + (num_pixels_y / threads.y));

    world **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(world *)));


    generate_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render_on_device<<<blocks, threads>>>(img.buffer,
                                          d_world,
                                          pixel00_loc, d_horizontal, d_vertical, camera_center,
                                          num_pixels_x, num_pixels_y);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    destroy_world<<<1, 1>>>(d_world);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_world));

    std::clog << "RayTracing::Done" << std::endl;

    double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    std::cout << "took " << seconds << " seconds" << std::endl;

    img.write_image("raytraced.png");
}
