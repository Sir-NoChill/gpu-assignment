#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cmath>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(err); \
        } \
    } while (0)

// Kernel for matrix multiplication with wavefront divergence
__global__ void matMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;
    float value_max = 10;

    // Divergence introduced by conditional accumulation
    for (int i = 0; i < N; ++i) {
        if (row < M && col < K) { // Diverging condition per thread
            value += A[row * N + i] * B[i * K + col]; 
        }
    }

    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

void matMul(const float* h_A, const float* h_B, float* h_C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_A, sizeA));
    HIP_CHECK(hipMalloc(&d_B, sizeB));
    HIP_CHECK(hipMalloc(&d_C, sizeC));

    // Copy host data to device
    HIP_CHECK(hipMemcpy(d_A, h_A, sizeA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, sizeB, hipMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    hipLaunchKernelGGL(matMulKernel, gridDim, blockDim, 0, 0, d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C, d_C, sizeC, hipMemcpyDeviceToHost));

    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

void readData(const std::string& filename, std::vector<float>& x, std::vector<float>& y) {
    std::ifstream infile(filename);
    float xi, yi;
    while (infile >> xi >> yi) {
        x.push_back(xi);
        y.push_back(yi);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::cout << "Reading data from " << argv[1] << std::endl;

    // Read input data
    std::vector<float> x, y;
    readData(argv[1], x, y);

    assert(x.size() == y.size());
    int N = x.size();

    // Prepare matrices for linear regression: X, Y, and XtX, XtY
    // X = [x 1], Xt = transpose(X), Y = y
    std::vector<float> X(2 * N);
    std::vector<float> Xt(2 * N);
    std::vector<float> Y(N);

    for (int i = 0; i < N; ++i) {
        X[i * 2] = x[i];
        X[i * 2 + 1] = 1.0f;
        Xt[i] = x[i];
        Xt[N + i] = 1.0f;
        Y[i] = y[i];
    }

    // XtX = Xt * X, XtY = Xt * Y
    float XtX[4] = {0};
    float XtY[2] = {0};
    matMul(Xt.data(), X.data(), XtX, 2, N, 2);
    matMul(Xt.data(), Y.data(), XtY, 2, N, 1);

    float epsilon = 1e-6; // Small value to ensure non-singularity
    XtX[0] += epsilon;
    XtX[3] += epsilon;

    // Solve XtX * [a b]^T = XtY for a and b (a = slope, b = intercept)
    // Using Cramer's rule since XtX is 2x2
    float det = XtX[0] * XtX[3] - XtX[1] * XtX[2];
    // if (fabs(det) < 1e-6) {
    //     std::cerr << "Matrix is singular, cannot solve for coefficients." << std::endl;
    //     return 1;
    // }

    float a = (XtY[0] * XtX[3] - XtY[1] * XtX[1]) / det;
    float b = (XtX[0] * XtY[1] - XtX[2] * XtY[0]) / det;

    std::cout << "Linear Regression Coefficients:" << std::endl;
    std::cout << "Slope (a): " << a << std::endl;
    std::cout << "Intercept (b): " << b << std::endl;

    return 0;
}

