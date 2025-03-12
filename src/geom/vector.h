#ifndef VECTOR_H
#define VECTOR_H
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <type_traits>

class vector {
    float coordinates[3];

public:
    __device__ __host__ vector() : coordinates{0, 0, 0} {}

    __device__ __host__ vector(const vector &other)     = default;
    __device__ __host__ vector(vector &&other) noexcept = default;

    __device__ __host__ vector(float x, float y, float z) : coordinates{x, y, z} {}

    __device__ __host__ vector &operator=(const vector &)     = default;
    __device__ __host__ vector &operator=(vector &&) noexcept = default;

    __device__ __host__ float x() const { return coordinates[0]; }
    __device__ __host__ float y() const { return coordinates[1]; }
    __device__ __host__ float z() const { return coordinates[2]; }

    __device__ __host__ float r() const { return coordinates[0]; }
    __device__ __host__ float g() const { return coordinates[1]; }
    __device__ __host__ float b() const { return coordinates[2]; }

    __device__ __host__ float &operator[](std::size_t index) { return coordinates[index]; }

    __device__ __host__ float operator[](std::size_t index) const {
        return coordinates[index];
    }

    __device__ __host__ float norm() const {
        float sum = 0;
        for (const auto &coord: coordinates) sum += coord * coord;
        return std::sqrt(sum);
    }

    __device__ __host__ vector operator+(const vector &other) const {
        return vector(coordinates[0] + other.coordinates[0],
                      coordinates[1] + other.coordinates[1],
                      coordinates[2] + other.coordinates[2]);
    }


    __device__ __host__ vector operator-(const vector &other) const {
        return vector(coordinates[0] - other.coordinates[0],
                      coordinates[1] - other.coordinates[1],
                      coordinates[2] - other.coordinates[2]);
    }

    __device__ __host__ vector operator*(float scalar) const {
        return vector(coordinates[0] * scalar, coordinates[1] * scalar, coordinates[2] * scalar);
    }
    __device__ __host__ friend vector operator*(float scalar, const vector &v) {
        return v * scalar;
    }

    __device__ __host__ vector operator/(float scalar) const {
        return vector(coordinates[0] / scalar, coordinates[1] / scalar, coordinates[2] / scalar);
    }

    __device__ __host__ vector operator+=(const vector &other) {
        coordinates[0] += other.coordinates[0];
        coordinates[1] += other.coordinates[1];
        coordinates[2] += other.coordinates[2];
        return *this;
    }

    __device__ __host__ vector operator-=(const vector &other) {
        coordinates[0] -= other.coordinates[0];
        coordinates[1] -= other.coordinates[1];
        coordinates[2] -= other.coordinates[2];
        return *this;
    }

    __device__ __host__ vector &operator*=(float scalar) {
        coordinates[0] *= scalar;
        coordinates[1] *= scalar;
        coordinates[2] *= scalar;
        return *this;
    }

    __device__ __host__ vector &operator/=(float scalar) {
        const float epsilon = 1e-8f;// Small number to avoid division by zero
        coordinates[0] /= (std::abs(scalar) > epsilon ? scalar : epsilon);
        coordinates[1] /= (std::abs(scalar) > epsilon ? scalar : epsilon);
        coordinates[2] /= (std::abs(scalar) > epsilon ? scalar : epsilon);

        return *this;
    }

    __device__ __host__ float dot(const vector &other) const {
        return coordinates[0] * other.coordinates[0] +
               coordinates[1] * other.coordinates[1] +
               coordinates[2] * other.coordinates[2];
    }

    __device__ __host__ vector cross(const vector &other) const {
        return vector(coordinates[1] * other.coordinates[2] - coordinates[2] * other.coordinates[1],
                      coordinates[2] * other.coordinates[0] - coordinates[0] * other.coordinates[2],
                      coordinates[0] * other.coordinates[1] - coordinates[1] * other.coordinates[0]);
    }

    __device__ __host__ vector unit() const & {
        auto magnitude = norm();
        return (magnitude == 0.0f) ? vector(0, 0, 0) : (*this) / magnitude;
    }


    friend std::ostream &operator<<(std::ostream &os, const vector &v) {
        return os << "(" << v.coordinates[0] << ", " << v.coordinates[1] << ", " << v.coordinates[2] << ")";
    }
};

using point3  = vector;
using vector3 = vector;
using color3  = vector;


#endif