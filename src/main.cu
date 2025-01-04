#include "geom/ray.h"
#include "geom/vector.h"
#include "image_buffer.h"
#include <algorithm>
#include <iostream>
#include <vector>

constexpr double aspect_ratio = 16.0 / 9.0;

constexpr int num_pixels_x = 1920;
constexpr int num_pixels_y = static_cast<int>(num_pixels_x / aspect_ratio);

constexpr float v_h = 2.0;
constexpr float v_w = v_h * (aspect_ratio);

constexpr float focal_length = 1.0;

const point3 origin(0, 0, 0);

const vector3 camera_center = origin;
const vector3 vertical(0, -v_h, 0);
const vector3 horizontal(v_w, 0, 0);

const vector3 Q            = camera_center - vector3(0, 0, focal_length) - vertical / 2 - horizontal / 2;
const vector3 d_vertical   = vertical / num_pixels_y;
const vector3 d_horizontal = horizontal / num_pixels_x;
const vector3 pixel00_loc  = Q + d_horizontal / 2 + d_vertical / 2;

__device__ __host__ float hit_sphere(vector3 const &C, float const r, ray const &R) {
    auto oc = C - R.origin();

    auto a = R.direction().dot(R.direction());
    auto b = -2 * R.direction().dot(oc);
    auto c = oc.dot(oc) - r * r;

    auto discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return -1.0f;
    return (-b - std::sqrt(discriminant)) / (2.0f * a);
}

__device__ __host__ color3 ray_color(ray const &R) {
    auto C = point3(0, 0, -1);
    auto t = hit_sphere(C, .5f, R);
    if (t > .0f) {// ray cuts through the sphere
        auto n = (R.at(t) - C).unit();
        return color3(n.x() + 1, n.y() + 1, n.z() + 1) * .5f;
    }

    auto unit = R.direction().unit();
    auto a    = 0.5f * (unit.y() + 1);

    return color3(.5, .7, 1) * a + color3(1, 1, 1) * (1 - a);
}

void render_on_host(color3 *buffer,
                    point3 const pixel00_loc, vector3 const dh, vector3 const dv, point3 const camera_center,
                    int max_x, int max_y) {
    for (int i = 0; i < max_x; i++) {
        for (int j = 0; j < max_y; j++) {
            auto pixel = pixel00_loc + (dv * float(j) / float(max_y)) + (dh * float(i) / float(max_x));
            ray R(camera_center, pixel - camera_center);
            int pixel_index     = j * max_x + i;
            buffer[pixel_index] = ray_color(R);
        }
    }
}

__global__ void render_on_device(color3 *buffer,
                                 point3 const pixel00_loc, vector3 const dh, vector3 const dv, point3 const camera_center,
                                 int max_x, int max_y) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    auto pixel = pixel00_loc + dv * j + dh * i;
    ray R(camera_center, pixel - camera_center);
    int pixel_index     = j * max_x + i;
    buffer[pixel_index] = ray_color(R);
}

int main(int argc, const char **argv) {
    auto img = ImgBuffer(num_pixels_x, num_pixels_y);

    clock_t start = clock();
#ifdef DEBUG
    std::clog << "RayTracing::Debug::init" << std::endl;

    render_on_host(img.buffer,
                   pixel00_loc, d_horizontal, d_vertical, camera_center,
                   num_pixels_x, num_pixels_y);

    std::clog << "RayTracing::Debug::done" << std::endl;
#else
    std::clog << "RayTracing::Init" << std::endl;
    dim3 threads(8, 8);
    dim3 blocks(1 + (num_pixels_x / threads.x), 1 + (num_pixels_y / threads.y));
    render_on_device<<<blocks, threads>>>(img.buffer,
                                          pixel00_loc, d_horizontal, d_vertical, camera_center,
                                          num_pixels_x, num_pixels_y);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    std::clog << "RayTracing::Done" << std::endl;
#endif
    double seconds = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
    std::cout << "took " << seconds << " seconds" << std::endl;

    img.write_image("raytraced.png");
}
