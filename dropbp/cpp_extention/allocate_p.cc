#include <torch/extension.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>

torch::Tensor allocate_p(torch::Tensor b, torch::Tensor C, torch::Tensor f, double target, double min_rate) {
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor!");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous!");
    TORCH_CHECK(C.device().is_cpu(), "C must be a CPU tensor!");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous!");
    TORCH_CHECK(f.device().is_cpu(), "w must be a CPU tensor!");
    TORCH_CHECK(f.is_contiguous(), "w must be contiguous!");

    std::priority_queue<std::pair<float, int64_t>> q;

    auto *b_data = b.data_ptr<float>();
    auto *C_data = C.data_ptr<float>();
    auto *f_data = f.data_ptr<float>();


    int64_t N = b.size(0);
    double f_sum = 0;
    for (int64_t i = 0; i < N; i++) {
        auto delta = -C_data[i]*(0.1+min_rate);
        q.push(std::make_pair(delta, i));
        f_sum += f_data[i];
    }

    f_sum *= 1-min_rate;

    while (f_sum > target) {        // Pick up the smallest increment (largest decrement)
        assert(!q.empty());
        auto delta = q.top().first;
        auto i = q.top().second;
        q.pop();
        b_data[i] += 0.1;
    
        f_sum -= f_data[i] * 0.1;
        if (b_data[i] < 0.9) {
            auto new_delta = delta-0.1*C_data[i];
            q.push(std::make_pair(new_delta, i));
        }
    }
    return b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allocate_p", &allocate_p, "allocate_p");
  }