#pragma once

#include <chrono>
#include <string>

/**
 * @brief Simple terminal progress bar with throughput reporting.
 */
class ProgressBar {
public:
    /**
     * @brief Creates a progress bar.
     * @param total Total number of work units.
     * @param label Text displayed before the progress information.
     * @param speed_divisor Divisor applied when calculating the displayed speed.
     * @param speed_unit Unit displayed after the speed value.
     */
    explicit ProgressBar(std::size_t total, std::string label = "", double speed_divisor = 1.0,
                         std::string speed_unit = "it/s");

    /**
     * @brief Sets the current progress value and redraws the bar.
     * @param current Number of completed work units.
     */
    void update(std::size_t current);
    /**
     * @brief Increases the current progress value.
     * @param amount Number of work units completed since the last update.
     */
    void increment(std::size_t amount = 1);
    /** @brief Marks the operation as complete and renders the final state. */
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