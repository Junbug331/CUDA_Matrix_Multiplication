#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>
#include <iterator>
#include <memory>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WARP_SIZE 32
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

template <typename T, int W = WARP_SIZE>
__global__ void kernel_MatMul_shared(T *a, T *b, T *c, int N, int M, int K)
{
    // Avoid bank conflict by padding
    __shared__ T sh_A[W][W+1];
    __shared__ T sh_B[W][W+1];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    T tmp = (T)0;

    // Note that thread dim are same i.e. (32, 32) 
    for (int offset=0; offset<K; offset+=W)
    {
        if (row < N && offset + tid_x < K)
            sh_A[tid_y][tid_x] = a[row * K + offset+ tid_x];
        else
            sh_A[tid_y][tid_x] = (T)0;

        if (offset + tid_y < K && col < M)
            sh_B[tid_y][tid_x] = b[offset*M + tid_y * M + col];
        else
            sh_B[tid_y][tid_x] = (T)0;

        //Wait for tiles to be loaded
        __syncthreads();

        // Matrix multiplication on tile(shared mem)
        for (int k=0; k<blockDim.x; ++k)
        {
            if (std::is_same<T, int>::value)
                tmp += sh_A[tid_y][k] * sh_B[k][tid_x];
            else
                tmp += __fmul_rn(sh_A[tid_y][k], sh_B[k][tid_x]);
        }

        __syncthreads();
    }

    if (row < N && col < M) c[row*M + col] = tmp;
}

template <typename T>
void matrixMul_shared(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, int N, int M, int K)
{
    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamCreate(&stream);

    cudaEventRecord(start, stream);

    // Device Matrix    
    auto deletor = [&](T *ptr){ cudaFree(ptr); };
    std::shared_ptr<T> d_a(new T[1], deletor);
    std::shared_ptr<T> d_b(new T[1], deletor);
    std::shared_ptr<T> d_c(new T[1], deletor);
    gpuAssert(cudaMalloc((void**) &d_a, sizeof(T)*N*K))
    gpuAssert(cudaMalloc((void**) &d_b, sizeof(T)*K*M))
    gpuAssert(cudaMalloc((void**) &d_c, sizeof(T)*N*M))

    int thread_num = WARP_SIZE;
    dim3 threads(thread_num, thread_num);

    int y_dim = (N + thread_num - 1)/thread_num;
    int x_dim = (M + thread_num - 1)/thread_num;
    dim3 blocks(x_dim, y_dim);


    cudaMemcpy(d_a.get(), a.data(), sizeof(T)*N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.get(), b.data(), sizeof(T)*K*M, cudaMemcpyHostToDevice);
    kernel_MatMul_shared<<<blocks, threads>>>(d_a.get(), d_b.get(), d_c.get(), N, M, K);
    gpuAssert(cudaGetLastError()) 
    gpuAssert(cudaMemcpy(c.data(), d_c.get(), sizeof(T)*N*M, cudaMemcpyDeviceToHost))

    cudaEventRecord(stop, stream);

    // Wait for the final event to be reached
    cudaEventSynchronize(stop);
    float millis;

    // Get the time between the start and stop event.
    cudaEventElapsedTime(&millis, start, stop);

    printf("tiledMatmul - event took %fms\n", millis);
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
    matrixMul_shared(a, b, c, N, M, K);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //verifyResult(a, b, c, N, M, K);


    return 0;
}