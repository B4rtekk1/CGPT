#include "utils/progress_bar.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>

ProgressBar::ProgressBar(const std::size_t total, std::string label, const double speed_divisor,
                         std::string speed_unit)
    : total_(total),
      label_(std::move(label)),
      speed_divisor_(speed_divisor),
      speed_unit_(std::move(speed_unit)),
      start_(std::chrono::steady_clock::now()),
      last_draw_(start_) {}

void ProgressBar::update(const std::size_t current) {
    current_ = std::min(current, total_);

    const auto now = std::chrono::steady_clock::now();
    if (now - last_draw_ >= std::chrono::milliseconds(100) || current_ == total_) {
        draw(now);
        last_draw_ = now;
    }
}

void ProgressBar::increment(const std::size_t amount) {
    update(current_ + amount);
}

void ProgressBar::finish() {
    const auto now = std::chrono::steady_clock::now();
    current_ = total_;
    draw(now);

    const double elapsed = std::chrono::duration<double>(now - start_).count();
    std::cout << " | Total time: " << std::fixed << std::setprecision(2)
              << elapsed << " s\n";
}

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

    std::cout << '\r';
    if (!label_.empty()) {
        std::cout << label_ << ' ';
    }
    std::cout << '['
              << std::string(filled, '=')
              << (filled < width ? ">" : "")
              << std::string(std::max(0, width - filled - 1), ' ')
              << "] " << std::fixed << std::setprecision(1)
              << progress * 100.0 << "% " << current_ << '/' << total_
              << " | " << std::setprecision(2) << speed << ' ' << speed_unit_
              << " | ETA: " << std::setprecision(1) << eta << " s"
              << std::flush;
}
