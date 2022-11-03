cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

project(CUDA_matrix_mul CUDA CXX)

find_package(spdlog REQUIRED)
if (spdlog_FOUND)
    message(STATUS "spdlog found - ${spdlog_DIR}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(baseline_matmul src/baseline_matmul.cu)
target_link_libraries(baseline_matmul -lnvToolsExt)

add_executable(tiled_matmul src/tiled_matmul.cu)
target_link_libraries(tiled_matmul -lnvToolsExt)

add_executable(joint_matmul src/joint_matmul.cu)
target_link_libraries(joint_matmul -lnvToolsExt)

add_executable(joint_matmul_overlap src/joint_matmul_overlap.cu)
target_link_libraries(joint_matmul_overlap PRIVATE 
    -lnvToolsExt
    spdlog::spdlog
    )
