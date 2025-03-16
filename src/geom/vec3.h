#ifndef VECTOR_H
#define VECTOR_H
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <type_traits>

class vec3 {
    float coordinates[3];

public:
    __device__ __host__ vec3() : coordinates{0, 0, 0} {}

    __device__ __host__ vec3(const vec3 &other)     = default;
    __device__ __host__ vec3(vec3 &&other) noexcept = default;

    __device__ __host__ vec3(float const x, float const y, float const z) : coordinates{x, y, z} {}

    __device__ __host__ vec3 &operator=(const vec3 &)     = default;
    __device__ __host__ vec3 &operator=(vec3 &&) noexcept = default;

    __device__ __host__ float x() const { return coordinates[0]; }
    __device__ __host__ float y() const { return coordinates[1]; }
    __device__ __host__ float z() const { return coordinates[2]; }

    __device__ __host__ float r() const { return coordinates[0]; }
    __device__ __host__ float g() const { return coordinates[1]; }
    __device__ __host__ float b() const { return coordinates[2]; }

    __device__ __host__ float &operator[](std::size_t const index) { return coordinates[index]; }

    __device__ __host__ float operator[](std::size_t const index) const {
        return coordinates[index];
    }

    __device__ __host__ float norm() const {
        float sum = 0;
        for (const auto &coord: coordinates) sum += coord * coord;
        return std::sqrt(sum);
    }

    __device__ __host__ vec3 operator+(const vec3 &other) const {
        return vec3(coordinates[0] + other.coordinates[0],
                      coordinates[1] + other.coordinates[1],
                      coordinates[2] + other.coordinates[2]);
    }


    __device__ __host__ vec3 operator-(const vec3 &other) const {
        return vec3(coordinates[0] - other.coordinates[0],
                      coordinates[1] - other.coordinates[1],
                      coordinates[2] - other.coordinates[2]);
    }


    template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    __device__ __host__ vec3 operator*(T const scalar) const {
        return vec3(coordinates[0] * scalar, coordinates[1] * scalar, coordinates[2] * scalar);
    }


    template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    __device__ __host__ friend vec3 operator*(T const scalar, const vec3 &v) {
        return v * scalar;
    }


    template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
    __device__ __host__ vec3 operator/(T const scalar) const {
        return vec3(coordinates[0] / scalar, coordinates[1] / scalar, coordinates[2] / scalar);
    }

    __device__ __host__ vec3 operator+=(const vec3 &other) {
        coordinates[0] += other.coordinates[0];
        coordinates[1] += other.coordinates[1];
        coordinates[2] += other.coordinates[2];
        return *this;
    }

    __device__ __host__ vec3 operator-=(const vec3 &other) {
        coordinates[0] -= other.coordinates[0];
        coordinates[1] -= other.coordinates[1];
        coordinates[2] -= other.coordinates[2];
        return *this;
    }

    __device__ __host__ vec3 &operator*=(float const scalar) {
        coordinates[0] *= scalar;
        coordinates[1] *= scalar;
        coordinates[2] *= scalar;
        return *this;
    }

    __device__ __host__ vec3 &operator/=(float const scalar) {
        constexpr float epsilon = 1e-8f;// Small number to avoid division by zero
        coordinates[0] /= std::abs(scalar) > epsilon ? scalar : epsilon;
        coordinates[1] /= std::abs(scalar) > epsilon ? scalar : epsilon;
        coordinates[2] /= std::abs(scalar) > epsilon ? scalar : epsilon;

        return *this;
    }

    __device__ __host__ float dot(const vec3 &other) const {
        return coordinates[0] * other.coordinates[0] +
               coordinates[1] * other.coordinates[1] +
               coordinates[2] * other.coordinates[2];
    }

    __device__ __host__ vec3 cross(const vec3 &other) const {
        return vec3(coordinates[1] * other.coordinates[2] - coordinates[2] * other.coordinates[1],
                      coordinates[2] * other.coordinates[0] - coordinates[0] * other.coordinates[2],
                      coordinates[0] * other.coordinates[1] - coordinates[1] * other.coordinates[0]);
    }

    __device__ __host__ vec3 unit() const & {
        auto const magnitude = norm();
        return magnitude == 0.0f ? vec3(0, 0, 0) : *this / magnitude;
    }


    friend std::ostream &operator<<(std::ostream &os, const vec3 &v) {
        return os << "(" << v.coordinates[0] << ", " << v.coordinates[1] << ", " << v.coordinates[2] << ")";
    }
};

using point3  = vec3;
using vector3 = vec3;
using color3  = vec3;


#endif