#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <memory>

#include <spdlog/spdlog.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvToolsExt.h>

#define TILE_SZ_A 64 
#define TILE_SZ_B 32 
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

using Clock = std::chrono::_V2::high_resolution_clock;

#define gpuAssert(code) \
    gpuErrChk(code, __FILE__, __LINE__)

inline void gpuErrChk(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPU assert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";

        if (abort)
            exit(code);
    }
}

template <typename T>
__global__ void print_matrix(T *a, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float tmp = (float)a[i * N + j];
            printf("%.2f ", tmp);
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T>
__global__ void print_matrix_col(T *a, int M, int N)
{
    for (int i = 0; i < N; ++i) // col
    {
        for (int j = 0; j < M; ++j) // row
        {
            float tmp = (float)a[j * N + i];
            printf("%.2f ", tmp);
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T>
__global__ void kernel_MatMul_joint(T *a, T *b, T *c, int m, int n, int k, int c_rows, int c_cols, int c_row_offset, int c_col_offset)
{
    /// verify answer fail when there is no parenthesis around _i
    /// b is transposed matrix
#define A(_i, _j) a[(_i)*k + (_j)]
#define B(_i, _j) b[(_j)*k + (_i)]
#define C(_i, _j) c[(_i)*c_cols + (_j)]

    // shared mem
    __shared__ T b_s[TILE_SZ_RATIO][TILE_SZ_B];

    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE_SZ_B; // offset

    // register
    T c_reg[TILE_SZ_B];
    for (unsigned int i = 0; i < TILE_SZ_B; ++i)
        c_reg[i] = (T)0;

    // loop over the input along k
    for (unsigned int tileidx = 0; tileidx < (k - 1) / TILE_SZ_RATIO + 1; ++tileidx)
    {
        // load b into tile
        const unsigned int i = threadIdx.y / TILE_SZ_B;
        const unsigned int j = threadIdx.y - (i * TILE_SZ_B);

        if (tileidx * TILE_SZ_RATIO + i < k && col + j < n)
            b_s[i][j] = B(tileidx * TILE_SZ_RATIO + i, col + j);
        else
            b_s[i][j] = 0.f;
        __syncthreads();

        for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx)
        {
            float a_reg;
            if (row < m && tileidx * TILE_SZ_RATIO + idx < k)
                a_reg = A(row, tileidx * TILE_SZ_RATIO + idx);
            else
                a_reg = 0.f;

            for (unsigned int outidx = 0; outidx < TILE_SZ_B; ++outidx)
                c_reg[outidx] += __fmul_rn(a_reg, b_s[idx][outidx]);
        }
        __syncthreads();
    }

    for (unsigned int outidx = 0; outidx < TILE_SZ_B; ++outidx)
    {
        if (c_row_offset + row < c_rows && c_col_offset + col + outidx < c_cols)
        {
            C((c_row_offset + row), (c_col_offset + col + outidx)) = c_reg[outidx];
        }
    }

#undef a
#undef b
#undef c
}

// check result on cpu
template <typename T = float>
void verifyResult(T *a, T *b, T *c, int M, int N, int K)
{
    // a: n x k
    // b: k x m
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            T tmp = static_cast<T>(0);
            for (int k = 0; k < K; ++k)
            {
                tmp += a[i * K + k] * b[k * N + j];
            }

            if (std::abs(tmp - c[i * N + j]) > 0.001)
            {
                std::cout << "wrong(" << i << ", " << j << "): " << tmp << ", " << c[i * N + j] << std::endl;
                return;
            }
        }
    }
    std::cout << "Success\n";
}

template <typename T>
void matrixMul_joint_overlap(int M, int N, int K)
{
    int n_chunk = 4;
    int a_chunk = (M + n_chunk - 1) / n_chunk;
    int b_chunk = (N + n_chunk - 1) / n_chunk;

    // initialize host data
    spdlog::info("generate host data");
    T *h_a, *h_b, *h_c;
    std::vector<T> tmp_b(K * N);
    gpuAssert(cudaHostAlloc((void **)&h_a, sizeof(T) * M * K, 0));
    gpuAssert(cudaHostAlloc((void **)&h_b, sizeof(T) * K * N, 0));
    gpuAssert(cudaHostAlloc((void **)&h_c, sizeof(T) * M * N, 0));

    auto randomNum = []()
    { return static_cast<T>(rand()) / static_cast<T>(RAND_MAX) * static_cast<T>(100); };
    std::generate(h_a, h_a + M * K, randomNum);
    std::generate(tmp_b.begin(), tmp_b.end(), randomNum);
    // traspose B
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_b[j * K + i] = tmp_b[i * N + j];

    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamCreate(&stream);

    cudaEventRecord(start, stream);

    // Allocate device data
    // spdlog::info("Allocate device data");
    std::vector<T *> d_a(n_chunk), d_b(n_chunk);
    T *d_c;
    for (int i = 0; i < n_chunk; ++i)
    {
        int a_lower = i * a_chunk;
        int b_lower = i * b_chunk;
        int a_upper = min(a_lower + a_chunk, M);
        int b_upper = min(b_lower + b_chunk, N);
        int a_width = a_upper - a_lower;
        int b_width = b_upper - b_lower;

        gpuAssert(cudaMalloc((void **)&d_a[i], sizeof(T) * a_width * K));
        gpuAssert(cudaMalloc((void **)&d_b[i], sizeof(T) * K * b_width));
    }
    gpuAssert(cudaMalloc((void **)&d_c, sizeof(T) * M * N));

    // Copy Data using multiple stream (host -> device)
    std::vector<cudaStream_t> copy_stream(n_chunk * 2);
    // spdlog::info("copying data(host -> device");
    for (int i = 0; i < n_chunk; ++i)
    {
        int a_lower = i * a_chunk;
        int b_lower = i * b_chunk;
        int a_upper = min(a_lower + a_chunk, M);
        int b_upper = min(b_lower + b_chunk, N);
        int a_width = a_upper - a_lower;
        int b_width = b_upper - b_lower;
        int i_a = i * 2;
        int i_b = i_a + 1;

        gpuAssert(cudaStreamCreate(&copy_stream[i_a]));
        gpuAssert(cudaStreamCreate(&copy_stream[i_b]));

        gpuAssert(cudaMemcpyAsync(&d_a[i][0], &h_a[a_lower * K], sizeof(T) * a_width * K, cudaMemcpyDefault, copy_stream[i_a]));
        gpuAssert(cudaMemcpyAsync(&d_b[i][0], &h_b[b_lower * K], sizeof(T) * b_width * K, cudaMemcpyDefault, copy_stream[i_b]));
    }
    gpuAssert(cudaDeviceSynchronize());

    // Execute Kernels in multiple streams
    // spdlog::info("Executing kernel");
    std::vector<cudaStream_t> kernel_stream(n_chunk * n_chunk);
    int idx = 0;
    for (int i = 0; i < n_chunk; ++i)
    {
        for (int j = 0; j < n_chunk; ++j)
        {
            int a_lower = i * a_chunk;
            int b_lower = j * b_chunk;
            int a_upper = min(a_lower + a_chunk, M);
            int b_upper = min(b_lower + b_chunk, N);
            int a_width = a_upper - a_lower;
            int b_width = b_upper - b_lower;

            dim3 block_dim(1, TILE_SZ_A);
            dim3 grid_dim((b_width + TILE_SZ_B - 1) / TILE_SZ_B, (a_width + TILE_SZ_A - 1) / TILE_SZ_A);

            gpuAssert(cudaStreamCreate(&kernel_stream[idx]));
            kernel_MatMul_joint<<<grid_dim, block_dim, 0, kernel_stream[idx]>>>(d_a[i], d_b[j], d_c, a_width, b_width, K, M, N, a_lower, b_lower);
            gpuAssert(cudaGetLastError());

            ++idx;
        }
    }
    gpuAssert(cudaDeviceSynchronize());

    // Copy data back to CPU
    // spdlog::info("copying data(device -> host)");
    for (int i = 0; i < n_chunk; ++i)
    {
        int lower = i * a_chunk;
        int upper = min(lower + a_chunk, M);
        int width = upper - lower;

        gpuAssert(cudaMemcpyAsync(&h_c[lower * N], &d_c[lower * N], sizeof(T) * width * N, cudaMemcpyDefault, copy_stream[i]));
    }
    gpuAssert(cudaDeviceSynchronize());

    // spdlog::info("Verifying result on CPU");
    verifyResult(h_a, tmp_b.data(), h_c, M, N, K);

    // Free Memory
    // spdlog::info("Deallocating memory");
    gpuAssert(cudaFreeHost(h_a));
    gpuAssert(cudaFreeHost(h_b));
    gpuAssert(cudaFreeHost(h_c));
    gpuAssert(cudaFree(d_c));
    for (int i = 0; i < n_chunk; ++i)
    {
        gpuAssert(cudaFree(d_a[i]));
        gpuAssert(cudaFree(d_b[i]));
    }

    cudaEventRecord(stop, stream);

    // Wait for the final event to be reached
    cudaEventSynchronize(stop);
    float millis;

    // Get the time between the start and stop event.
    cudaEventElapsedTime(&millis, start, stop);

    printf("joint Matmul overlap - event took %fsec\n", millis / 1000);
}

int main()
{
    int M, N, K;
    N = 103;
    M = 121;
    K = 107;

    matrixMul_joint_overlap<float>(M, N, K);

    return 0;
}