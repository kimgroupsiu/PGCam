#ifndef _BLOBIMAGEPROCESS_KERNEL_H_
#define _BLOBIMAGEPROCESS_KERNEL_H_

#include "cuda_kernels.h"
#define th_sz 32

__inline__ __device__ float3 warp_reduce_sum_triple(float3 val) {
    for (int offset = th_sz/2; offset > 0; offset /= 2) {
        val.x += __shfl_xor_sync(0xffffffff, val.x, offset);
        val.y += __shfl_xor_sync(0xffffffff, val.y, offset);
        val.z += __shfl_xor_sync(0xffffffff, val.z, offset);
    }
    return val;
}
__inline__ __device__ float warp_reduce_sum_float(float val) {
    for (int offset = th_sz/2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}



__global__ void compute_centroid(float* __restrict__ output, const float* __restrict__ input) {
    output[0] = input[0] / input[2]; // center x
    output[1] = input[1] / input[2]; // center x
    output[3] = input[2] / 4; // size
    float mu_x, mu_y, mu_xy;
    mu_x = input[3] / input[2] - output[0] * output[0];
    mu_y = input[4] / input[2] - output[1] * output[1];
    mu_xy = input[5] / input[2] - output[0] * output[1];
    output[2] = atan2f(2 * mu_xy, mu_x - mu_y) / 2; // orientation
}



__global__ void ImageMoment_binarization_kernel(
        float* __restrict__ output,
        float* __restrict__ input,
        const int2 input_size,
        const float threshold,
        float * __restrict__ d_save
        )
{
    static __shared__ float shared[th_sz*6]; // warp size
    int lane = threadIdx.x;
    int wid = threadIdx.y;
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int gridId = blockIdx.y * gridDim.x + blockIdx.x;
    int gridSz = gridDim.x*gridDim.y;


    float3 g = { 0 };
    float3 h = { 0 };

    for (int y = idx_y; y < input_size.y; y += blockDim.y * gridDim.y) {
        for (int x = idx_x; x < input_size.x; x += blockDim.x * gridDim.x) {
            int gid = y*input_size.x + x;
            d_save[gid] = 0.0f;
            if (input[gid] > threshold) {
                g.x += x;
                g.y += y;
                g.z += 1;
                h.x += x*x;
                h.y += y*y;
                h.z += x*y;
                d_save[gid] = 255.0f;
            }
        }
    }

    float3 g_sum = warp_reduce_sum_triple(g);
    float3 h_sum = warp_reduce_sum_triple(h);

    if (lane == 0) {
        shared[wid] = g_sum.x;
        shared[wid + th_sz] = g_sum.y;
        shared[wid + 2*th_sz] = g_sum.z;
        shared[wid + 3*th_sz] = h_sum.x;
        shared[wid + 4*th_sz] = h_sum.y;
        shared[wid + 5*th_sz] = h_sum.z;
    }
    __syncthreads();
    if (wid == 0) {
        float shared_sum1 = warp_reduce_sum_float(shared[lane]);
        float shared_sum2 = warp_reduce_sum_float(shared[lane + th_sz]);
        float shared_sum3 = warp_reduce_sum_float(shared[lane + 2*th_sz]);
        float shared_sum4 = warp_reduce_sum_float(shared[lane + 3*th_sz]);
        float shared_sum5 = warp_reduce_sum_float(shared[lane + 4*th_sz]);
        float shared_sum6 = warp_reduce_sum_float(shared[lane + 5*th_sz]);
        if (lane == 0) {
            output[gridId] = shared_sum1;
            output[gridSz + gridId] = shared_sum2;
            output[gridSz * 2 + gridId] = shared_sum3;
            output[gridSz * 3 + gridId] = shared_sum4;
            output[gridSz * 4 + gridId] = shared_sum5;
            output[gridSz * 5 + gridId] = shared_sum6;
        }
    }

}

extern "C" void getImageMoment_GPU(
    float* output, // 1024 float size
    float* input, // input image
    int width, int height,
    float threshold,
    float* d_save // output binary image
    ) {
    dim3 _threads(th_sz, th_sz);
    dim3 _blocks(1, 1);
    int2 i_size = { width, height };
    ImageMoment_binarization_kernel << <_blocks, _threads >> >(output, input, i_size, threshold, d_save);
    compute_centroid<< <1, 1>> > (output + 6, output);
};











#endif
