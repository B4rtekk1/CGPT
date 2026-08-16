#pragma once

#include <chrono>
#include <string>

class ProgressBar {
public:
    explicit ProgressBar(std::size_t total, std::string label = "", double speed_divisor = 1.0,
                         std::string speed_unit = "it/s");

    void update(std::size_t current);
    void increment(std::size_t amount = 1);
    void finish();

private:
    void draw(std::chrono::steady_clock::time_point now) const;

    std::size_t total_;
    std::size_t current_ = 0;
    std::string label_;
    double speed_divisor_;
    std::string speed_unit_;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_draw_;
};
