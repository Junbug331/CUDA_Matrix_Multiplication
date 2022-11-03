#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>
#include <iterator>
#include <memory>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define gpuAssert(code) {gpuErrChk(code, __FILE__, __LINE__);}

void inline gpuErrChk(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPU assert: " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// check result on CPU
template<typename T>
void verifyResult(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, int N, int M, int K)
{
    // A: N x K
    // B: K x M
    for (int i=0; i<N; ++i)
    {
        for (int j=0; j<M; ++j)
        {
            T tmp = (T)0;
            for (int k=0; k<K; ++k)
            {
                tmp += a[i*K + k] * b[k*M + j];
            }

            if (std::abs(tmp - c[i*M + j]) > std::numeric_limits<T>::epsilon())
            {
                std::cout << "wrong: " << tmp << ", " << c[i*M + j] << std::endl;
                return;
            }
            //assert(tmp == c[i*M + j]);
        }
    }

    std::cout << "Success\n";
}


template<typename T>
__global__ void kernel_MatMul(T *a, T *b, T *c, int N, int M, int K)
{
    int gid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;
    int x, y, k;

    T tmp = (T)0;

    for (y=gid_y; y<N; y+=stride_y)
    {
        for (x=gid_x; x<M; x+=stride_x)
        {
            tmp = 0;
            for (k=0; k<K; ++k)
            {
                if (std::is_same<T, int>::value)
                    tmp += a[y*K + k] * b[k*M + x];
                else
                    tmp += __fmul_rn(a[y*K + k], b[k*M + x]);
            }
            c[y*M + x] = tmp;
        }
    }
}

template <typename T>
void matrixMul_baseline(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, int N, int M, int K)
{
    T *d_a, *d_b, *d_c;
    gpuAssert(cudaMalloc(&d_a, sizeof(T)*N*K))
    gpuAssert(cudaMalloc(&d_b, sizeof(T)*K*M))
    gpuAssert(cudaMalloc(&d_c, sizeof(T)*N*M))

    gpuAssert(cudaMemcpy(d_a, a.data(), sizeof(T)*N*K, cudaMemcpyHostToDevice))
    gpuAssert(cudaMemcpy(d_b, b.data(), sizeof(T)*K*M, cudaMemcpyHostToDevice))


    int num_threads = 32;
    int num_blocks_y = (N + num_threads - 1) / num_threads;
    int num_blocks_x = (M + num_threads - 1) / num_threads;

    dim3 threads(num_threads, num_threads);
    dim3 blocks(num_blocks_x, num_blocks_y);

    kernel_MatMul<<<blocks, threads>>>(d_a, d_b, d_c, N, M, K);
    cudaDeviceSynchronize();
    gpuAssert(cudaGetLastError())

    gpuAssert(cudaMemcpy(c.data(), d_c, sizeof(T)*N*M, cudaMemcpyDeviceToHost))

    gpuAssert(cudaFree(d_a))
    gpuAssert(cudaFree(d_b))
    gpuAssert(cudaFree(d_c))
}

int main() 
{
    int N, M, K;
    N = 10000;
    M = 10200;
    K = 10009;

    std::vector<float> a(N*K), b(K*M), c(N*M);
    std::generate(a.begin(), a.end(), [](){return 100 * static_cast<float>(rand() / static_cast<float>(RAND_MAX));});
    std::generate(b.begin(), b.end(), [](){return 100 * static_cast<float>(rand() / static_cast<float>(RAND_MAX));});

    auto start = std::chrono::steady_clock::now();    
    matrixMul_baseline(a, b, c, N, M, K);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "baseline took " << elapsed << "ms\n";
    //verifyResult(a, b, c, N, M, K);
    return 0;
}

