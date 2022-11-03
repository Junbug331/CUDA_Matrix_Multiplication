#include <iostream>
#include <algorithm>
#include <cassert>
#include <vector>
#include <iterator>
#include <memory>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SZ_A 64
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

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

template <typename T>
__global__ void kernel_MatMul_joint(T *a, T *b, T *c, int M, int N, int K)
{
    // A: MxK, B: KxN, C: MxN
#define A(_i, _j) a[(_i)*K + _j]
#define B(_i, _j) b[(_i)*N + _j]
#define C(_i, _j) c[(_i)*N + _j]

    // Shared memory for tiling input B array
    // Dimension of a block is 64 x 1 which can be rearranged as 4 x 16
    __shared__ float B_s[TILE_SZ_RATIO][TILE_SZ_B];

    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE_SZ_B;

    T c_reg[TILE_SZ_B]; // This will account for lack of dimension of blockDim.x which is 1

    // Initialize output values
    for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx)
        c_reg[outIdx] = T(0);

    // Loop over the input files
    for (unsigned int tileIdx = 0; tileIdx < (K-1)/TILE_SZ_RATIO + 1; ++ tileIdx)
    {
        // Load the tile of B into shared memory
        // 64 x 1 -> 4 x 16  
        const unsigned int i = threadIdx.y / TILE_SZ_B;
        const unsigned int j = threadIdx.y - i*TILE_SZ_B; 

        // Edge case for B (K x N)
        // tileIdx * TILE_SZ_RATIO -> offset
        // offset + i: row
        // col + j: colum
        if (tileIdx * TILE_SZ_RATIO + i < K && col + j < N)
            B_s[i][j] = B(tileIdx*TILE_SZ_RATIO + i, col + j);
        else
            B_s[i][j] = (T)0;
        __syncthreads();

        // Loop over elements inside the tile
        for (unsigned int idx = 0; idx < TILE_SZ_RATIO; ++idx)
        {
            // Load tile of A matrix into register
            float a_reg;
            // Edge case for A (M x K)
            if (row < M && tileIdx * TILE_SZ_RATIO + idx < K)
                a_reg = A(row, tileIdx*TILE_SZ_RATIO + idx);
            else
                a_reg = (T)0;
            
            // output calculation
            for (unsigned int outIdx = 0; outIdx < TILE_SZ_B; ++outIdx)
            {
                if (std::is_same<T, int>::value)
                    c_reg[outIdx] += a_reg * B_s[idx][outIdx];
                else
                    c_reg[outIdx] += __fmul_rn(a_reg, B_s[idx][outIdx]);
            }
        }
        __syncthreads();
    }

    for (unsigned int outIdx = 0; outIdx <TILE_SZ_B; ++outIdx)
    {
        if (row < M && col + outIdx < N)  
            C(row, col + outIdx) = c_reg[outIdx];
    }
}

template <typename T>
void matrixMul_joint(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c, int M, int N, int K)
{
    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamCreate(&stream);

    cudaEventRecord(start, stream);

    auto deleter = [&](T* ptr){ cudaFree(ptr); };
    std::shared_ptr<T> d_a(new T[1], deleter);
    std::shared_ptr<T> d_b(new T[1], deleter);
    std::shared_ptr<T> d_c(new T[1], deleter);

    // Allocate Memory
    gpuAssert(cudaMalloc((void**)&d_a, sizeof(T)*M*K))
    gpuAssert(cudaMalloc((void**)&d_b, sizeof(T)*K*N))
    gpuAssert(cudaMalloc((void**)&d_c, sizeof(T)*M*N))

    // Copy to device
    gpuAssert(cudaMemcpy(d_a.get(), a.data(), sizeof(T)*M*K, cudaMemcpyHostToDevice))
    gpuAssert(cudaMemcpy(d_b.get(), b.data(), sizeof(T)*K*N, cudaMemcpyHostToDevice))

    dim3 grid_dim((N+TILE_SZ_B-1)/TILE_SZ_B, (M+TILE_SZ_A-1)/TILE_SZ_A);
    dim3 block_dim(1, TILE_SZ_A);
    gpuAssert(cudaGetLastError())
    kernel_MatMul_joint<<<grid_dim, block_dim>>>(d_a.get(), d_b.get(), d_c.get(), M, N, K);
    gpuAssert(cudaGetLastError())

    // Copy to Host
    gpuAssert(cudaMemcpy(c.data(), d_c.get(), sizeof(T)*M*N, cudaMemcpyDeviceToHost))

    cudaEventRecord(stop, stream);

    // Wait for the final event to be reached
    cudaEventSynchronize(stop);
    float millis;

    // Get the time between the start and stop event.
    cudaEventElapsedTime(&millis, start, stop);

    printf("joint Matmul - event took %fsec\n", millis/1000);
}

int main() 
{
    int N, M, K;
    N = 1000;
    M = 1200;
    K = 1009;

    std::vector<float> a(N*K), b(K*M), c(N*M);
    std::generate(a.begin(), a.end(), [](){return 100 * static_cast<float>(rand() / static_cast<float>(RAND_MAX));});
    std::generate(b.begin(), b.end(), [](){return 100 * static_cast<float>(rand() / static_cast<float>(RAND_MAX));});

    auto start = std::chrono::steady_clock::now();    
    matrixMul_joint(a, b, c, N, M, K);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //verifyResult(a, b, c, N, M, K);
    return 0;
}

