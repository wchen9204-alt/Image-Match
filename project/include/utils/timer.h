#pragma once

#include <chrono>

namespace ir {

// ---------------------------------------------------------------------------
// ScopedTimer：作用域计时器，析构时写入耗时毫秒数。
// ---------------------------------------------------------------------------
class Timer {
public:
    using clock      = std::chrono::steady_clock;
    using time_point = clock::time_point;

    Timer() : start_(clock::now()) {}

    void reset() { start_ = clock::now(); }

    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(clock::now() - start_).count();
    }

private:
    time_point start_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(double& out_ms) : out_(out_ms), t_() {}
    ~ScopedTimer() { out_ = t_.elapsedMs(); }

    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double& out_;
    Timer   t_;
};

} // namespace ir
