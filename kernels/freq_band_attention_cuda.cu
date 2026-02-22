/*
 * freq_band_attention_cuda.cu — CUDA C++ kernel stub for frequency-band attention.
 *
 * This file provides the CUDA kernel source. It can be compiled via
 * torch.utils.cpp_extension when CUDA is available.
 *
 * Build:
 *   python -c "from torch.utils.cpp_extension import load; \
 *     load('fba_cuda', sources=['kernels/freq_band_attention_cuda.cu'])"
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: add row + column band biases to attention scores
__global__ void fba_bias_kernel(
    float* __restrict__ scores,   // (N, N) per (B, H) slice
    const float* __restrict__ bias_q,  // (N,)
    const float* __restrict__ bias_k,  // (N,)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;

    int row = idx / N;
    int col = idx % N;

    scores[idx] += bias_q[row] + bias_k[col];
}

// C++ wrapper
torch::Tensor fba_add_bias(
    torch::Tensor scores,   // (B, H, N, N)
    torch::Tensor bias_q,   // (B, H, N)
    torch::Tensor bias_k    // (B, H, N)
) {
    auto B = scores.size(0);
    auto H = scores.size(1);
    auto N = scores.size(2);

    auto output = scores.clone();

    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            fba_bias_kernel<<<blocks, threads>>>(
                output[b][h].data_ptr<float>(),
                bias_q[b][h].data_ptr<float>(),
                bias_k[b][h].data_ptr<float>(),
                N
            );
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fba_add_bias", &fba_add_bias,
          "Frequency-band attention bias addition (CUDA)");
}
