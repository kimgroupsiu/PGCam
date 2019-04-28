#include "dsp_gpu.h"
#include "cuda_kernels.h"
#include <stdlib.h>
#include <math.h>


DSP_gpu::DSP_gpu()
{
    cudaError_t err = cudaDeviceReset();
    Src_8U = NULL;
    // device
    d_Src_8U = NULL;
    d_gImg = NULL; // gaussian image
    d_dummy = NULL; // intermediate images: gaussian_x,
    d_gKernel = NULL; gKernel_radius = 0;
    d_data = NULL; // image momentum data(1024x1)

    threshold = 60.0f;


}


DSP_gpu::~DSP_gpu()
{
    if (d_Src_8U) cudaFree(d_Src_8U);
    if (d_gKernel) cudaFree(d_gKernel);
    if (d_gImg) cudaFree(d_gImg);
    if (d_dummy) cudaFree(d_dummy);
    if (d_data) cudaFree(d_data);
}

cudaError_t DSP_gpu::init_memset(int w, int h) {
    cudaError_t err;
    width = w; height = h;
    bytePerIm = w*h*sizeof(unsigned char);
    err = cudaMalloc((void **)(&d_Src_8U), bytePerIm);
    err = cudaMalloc((void **)(&d_gImg), bytePerIm*sizeof(float));
    err = cudaMalloc((void **)(&d_dummy), bytePerIm*sizeof(float));
    err = cudaMalloc((void **)(&d_data), 1024*sizeof(float));

    err = cudaMemset((unsigned char*)d_Src_8U, 0, bytePerIm);
    err = cudaMemset((unsigned char*)d_gImg, 0, bytePerIm*sizeof(float));
    err = cudaMemset((unsigned char*)d_dummy, 0, bytePerIm*sizeof(float));
    err = cudaMemset((unsigned char*)d_data, 0, 1024*sizeof(float));

    filterCreation(5, 3.0);

    return err;
}

void DSP_gpu::proc(unsigned char * src) {
    cudaError_t err;
    upload_src8U(src); // upload
    imfilter_gaussian(d_gImg); // gauissian filter
    getImageMoment_GPU(d_data, d_gImg, (int)width, (int)height, threshold, d_dummy); // compute centroid

    float A[10]={0};
    err = cudaMemcpy(A, d_data, 10*sizeof(float), cudaMemcpyDeviceToHost);
    pos.x = A[6]; pos.y = A[7]; pos.z = A[8]; pos.w = A[9];
    //copyfloat2uchar_gpu(d_Src_8U, d_dummy, height*width);
    gpuErrchk( cudaPeekAtLastError() );
}

cudaError_t DSP_gpu::upload_src8U(unsigned char * src) {
    cudaError_t err;
    err = cudaMemcpy((unsigned char*)d_Src_8U, (const unsigned char*)src, bytePerIm, cudaMemcpyHostToDevice);
    copyuchar2float_gpu((float*)d_gImg, (unsigned char*)d_Src_8U, height*width);
    return err;
}

void DSP_gpu::imfilter_gaussian(float * d_src)
{
    convolutionRowsGPU(d_dummy, d_src, (int)width, (int)height, d_gKernel, (int)gKernel_radius);
    convolutionColumnsGPU(d_gImg, d_dummy,(int)width, (int)height, d_gKernel, (int)gKernel_radius);
}


cudaError_t DSP_gpu::testfn(unsigned char * dst_8u)
{
    cudaError_t err;
    err = cudaMemcpy(dst_8u, d_Src_8U, bytePerIm, cudaMemcpyDeviceToHost);
    return err;
}


void DSP_gpu::setThreshold(float _th) {
    if (_th < 0) _th = 0;
    if (_th > 255) _th = 255;
    threshold = _th;
}
float4 DSP_gpu::getPos(void) {
    return pos;
}

void DSP_gpu::filterCreation(int r, float sig) // Function to create Gaussian filter
{
    // variable & memory setting
    gKernel_radius = r;
    int len = 2*gKernel_radius +1;
    sigma = sig;
    float * gKernel = (float *)malloc(len * sizeof(float));

    /* Gaussian Filter Creation */
    // intialising standard deviation to 1.0
    float s = 2.0 * sigma * sigma;
    // sum is for normalization
    float sum = 0.0;
    // generating 5x5 kernel
    for (int x = -gKernel_radius; x <= gKernel_radius; x++) {
        gKernel[x + gKernel_radius] = (exp(-(x * x) / s)) / (3.141592 * s);
        sum += gKernel[x + gKernel_radius];
    }
    // normalising the Kernel
    for (int i = 0; i < len; ++i)
        gKernel[i] /= sum;

    // upload to GPU
    cudaError_t err;
    if (d_gKernel) cudaFree(d_gKernel);
    err = cudaMalloc((void **)(&d_gKernel), len * sizeof(float));
    err = cudaMemcpy((float*)d_gKernel, (const float*)gKernel, len * sizeof(float), cudaMemcpyHostToDevice);
    free(gKernel);
}
