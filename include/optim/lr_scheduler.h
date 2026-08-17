#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

/** @brief Configuration for a warmup followed by cosine learning-rate decay. */
struct LearningRateSchedulerOptions {
    float initial_learning_rate = 1.0e-4F;
    float minimum_learning_rate = 0.0F;
    std::size_t warmup_steps = 0;
    std::size_t total_steps = 1;
};

/**
 * @brief Computes the learning rate for a zero-based optimizer step.
 *
 * During warmup the rate grows linearly from one warmup fraction of the
 * initial rate to the initial rate. It then follows a cosine curve and
 * reaches the minimum rate on the last configured step.
 */
class LearningRateScheduler {
public:
    explicit LearningRateScheduler(LearningRateSchedulerOptions options)
        : options_(options) {
        if (!std::isfinite(options_.initial_learning_rate) ||
            !std::isfinite(options_.minimum_learning_rate) ||
            options_.initial_learning_rate <= 0.0F ||
            options_.minimum_learning_rate < 0.0F ||
            options_.minimum_learning_rate > options_.initial_learning_rate ||
            options_.total_steps == 0 || options_.warmup_steps >= options_.total_steps)
            throw std::invalid_argument("Invalid learning-rate scheduler options");
    }

    [[nodiscard]] float learning_rate(const std::size_t step) const noexcept {
        if (step < options_.warmup_steps) {
            const float progress = static_cast<float>(step + 1) /
                static_cast<float>(options_.warmup_steps);
            return options_.initial_learning_rate * progress;
        }
        if (step >= options_.total_steps - 1) return options_.minimum_learning_rate;

        const std::size_t decay_steps = options_.total_steps - options_.warmup_steps;
        const float progress = static_cast<float>(step - options_.warmup_steps) /
            static_cast<float>(std::max<std::size_t>(1, decay_steps - 1));
        constexpr float pi = 3.14159265358979323846F;
        const float cosine = 0.5F * (1.0F + std::cos(pi * progress));
        return options_.minimum_learning_rate +
               (options_.initial_learning_rate - options_.minimum_learning_rate) * cosine;
    }

private:
    LearningRateSchedulerOptions options_;
};
