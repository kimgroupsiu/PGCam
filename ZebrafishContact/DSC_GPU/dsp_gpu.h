#ifndef DSP_GPU_H
#define DSP_GPU_H

#include "cuda_kernels.h"

class DSP_gpu
{
public:
    DSP_gpu();
    ~DSP_gpu();

    /* functions */
    cudaError_t init_memset(int w, int h);
    void proc(unsigned char * src_8u);

    cudaError_t upload_src8U(unsigned char * src_8u);
    void imfilter_gaussian(float * src);
    void filterCreation(int r, float sig);

    void setThreshold(float _th);
    float4 getPos(void);
    cudaError_t testfn(unsigned char * dst_8u);
private:
    /* host(cpu) variable setting */
    unsigned char * Src_8U;
    int width; int height; // Image size
    size_t bytePerIm; size_t bytePerIm_float; // Image size in byte
    int gKernel_radius; float sigma;
    /* device(gpu) variable settings */
    unsigned char * d_Src_8U; // Source copyto
    float * d_gKernel; // gaussian Kernel
    float * d_gImg; // gaussian image
    float * d_data; // image momentum data(1024x1)
    float * d_dummy; // intermediate images: gaussian_x, binary image
    /* Image processing variable settings */
    float threshold;
    float4 pos; // x, y, th, size
};

#endif // DSP_GPU_H
