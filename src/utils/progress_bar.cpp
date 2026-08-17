#include "utils/progress_bar.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>

/**
 * @file 1775283d-b192-4176-a5e6-8ee7a6dbef26.cpp
 * @brief Implementation of the terminal progress bar.
 */

/**
 * @brief Initializes a progress bar and starts its timer.
 * @param total Total number of work units.
 * @param label Text displayed before the progress information.
 * @param speed_divisor Divisor applied to the calculated throughput.
 * @param speed_unit Unit displayed after the throughput value.
 */
ProgressBar::ProgressBar(const std::size_t total, std::string label, const double speed_divisor,
                         std::string speed_unit)
    : total_(total),
      label_(std::move(label)),
      speed_divisor_(speed_divisor),
      speed_unit_(std::move(speed_unit)),
      start_(std::chrono::steady_clock::now()),
      last_draw_(start_) {}

/**
 * @brief Updates the current progress and redraws when needed.
 *
 * Progress is clamped to the configured total. Redrawing is throttled to at
 * most once every 100 milliseconds unless the operation has completed.
 *
 * @param current Number of completed work units.
 */
void ProgressBar::update(const std::size_t current) {
    current_ = std::min(current, total_);

    const auto now = std::chrono::steady_clock::now();
    if (now - last_draw_ >= std::chrono::milliseconds(100) || current_ == total_) {
        draw(now);
        last_draw_ = now;
    }
}

/**
 * @brief Advances the progress by a specified number of work units.
 * @param amount Number of completed work units to add.
 */
void ProgressBar::increment(const std::size_t amount) {
    update(current_ + amount);
}

void ProgressBar::set_suffix(std::string suffix) {
    suffix_ = std::move(suffix);
    const auto now = std::chrono::steady_clock::now();
    draw(now);
    last_draw_ = now;
}

/**
 * @brief Completes the progress bar and prints the total elapsed time.
 */
void ProgressBar::finish() {
    const auto now = std::chrono::steady_clock::now();
    current_ = total_;
    draw(now);

    const double elapsed = std::chrono::duration<double>(now - start_).count();
    std::cout << " | Total time: " << std::fixed << std::setprecision(2)
              << elapsed << " s\n";
}

/**
 * @brief Renders the current progress, throughput, and estimated time remaining.
 * @param now Timestamp used for elapsed-time and speed calculations.
 */
void ProgressBar::draw(const std::chrono::steady_clock::time_point now) const {
    constexpr int width = 40;
    const double progress = total_ == 0
                                 ? 1.0
                                 : static_cast<double>(current_) / static_cast<double>(total_);
    const int filled = static_cast<int>(progress * width);
    const double elapsed = std::chrono::duration<double>(now - start_).count();
    const double speed = elapsed > 0.0
                             ? static_cast<double>(current_) / elapsed / speed_divisor_
                             : 0.0;
    const double eta = speed > 0.0
                           ? static_cast<double>(total_ - current_) / speed
                           : 0.0;

    // Clear the prior render first: a new suffix can be shorter than the old
    // one, and metrics must not leave stale characters in the terminal.
    std::cout << "\x1b[2K\r";
    if (!label_.empty()) {
        std::cout << label_ << ' ';
    }
    std::cout << '['
              << std::string(filled, '=')
              << std::string(std::max(0, width - filled), ' ')
              << "] " << std::fixed << std::setprecision(1)
              << progress * 100.0 << "% " << current_ << '/' << total_
              << " | " << std::setprecision(2) << speed << ' ' << speed_unit_
              << " | ETA: " << std::setprecision(1) << eta << " s"
              << suffix_
              << std::flush;
}
