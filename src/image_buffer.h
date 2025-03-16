#ifndef IMAGE_H
#define IMAGE_H

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include "util.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

class ImgBuffer {
public:
    int width, height;
    int num_pixels;
    color3 *buffer;

    ImgBuffer(int const w, int const h) : width(w), height(h), num_pixels(w * h) {


#ifdef DEBUG
        std::clog << "RayTracer::Debug::Host memory alloc" << std::endl;
        buffer = new color3[num_pixels];
#else
        auto const buffer_size = num_pixels * sizeof(color3);
        std::clog << "rayTracer::cuda memory alloc" << std::endl;
        checkCudaErrors(cudaMallocManaged(reinterpret_cast<void **>(&buffer), buffer_size));
        checkCudaErrors(cudaMemset(buffer, 0, buffer_size));
#endif
    }

    ~ImgBuffer() {
#ifdef DEBUG
        std::clog << "RayTracer::Debug::Host memory dealloc" << std::endl;
        delete[] buffer;
#else
        std::clog << "rayTracer::cuda memory dealloc" << std::endl;
        checkCudaErrors(cudaFree(buffer));
#endif
    }

    void write_image(const std::string &filename) const {
        std::vector<unsigned char> output_buffer(num_pixels * 3);
        for (size_t i = 0; i < num_pixels; ++i) {
            output_buffer[3 * i]     = static_cast<unsigned char>(std::clamp(buffer[i].r() * 255.0f, 0.0f, 255.0f));
            output_buffer[3 * i + 1] = static_cast<unsigned char>(std::clamp(buffer[i].g() * 255.0f, 0.0f, 255.0f));
            output_buffer[3 * i + 2] = static_cast<unsigned char>(std::clamp(buffer[i].b() * 255.0f, 0.0f, 255.0f));
        }
        if (!stbi_write_png(filename.c_str(), width, height, 3, output_buffer.data(), width * 3))
            std::cerr << "ERROR: Failed to write image!" << std::endl;
    }
};

#endif