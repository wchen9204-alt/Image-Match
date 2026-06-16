#pragma once

#include <chrono>

namespace ir {

/// 简单的单段计时器。
class Timer {
public:
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;

    /// 构造后立即开始计时。
    Timer() : _start(clock::now()) {}

    /// 重新开始计时。
    void reset() { _start = clock::now(); }

    /// 返回从上次 reset 或构造以来的毫秒数。
    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(clock::now() - _start).count();
    }

private:
    time_point _start;
};

/// 作用域计时器，析构时把耗时写入外部变量。
class ScopedTimer {
public:
    /// 构造时开始计时，析构时写回结果。
    explicit ScopedTimer(double& out_ms) : _out(out_ms), _t() {}

    /// 析构时写入累计耗时。
    ~ScopedTimer() { _out = _t.elapsedMs(); }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double& _out;
    Timer _t;
};

} // namespace ir

