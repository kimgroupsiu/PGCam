#ifndef CUDA_KERNELS
#define CUDA_KERNELS

#define GPU0 1
#define threadsPerBlock 256

#include <cuda_runtime.h>
#include <stdio.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//char gpuErrorMsg[1024];
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

    //char outputFilename[] = "d:\\TrackingMicroscopeData\\_report\\gpuErrorMsg.txt";
    //FILE  * ofp = fopen(outputFilename, "w");
    //fprintf(ofp,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    //fclose(ofp);
    //if (abort) exit(code);
   }
}



extern "C" void VecAdd_kernelGPU(
    float *d_Src1,
    float *d_Src2,
    float *d_Dst,
    int numElements
);


extern "C" void copyfloat2uchar_gpu(
    unsigned char * d_dst,
    float * d_Src,
    int numElements
);


extern "C" void copyuchar2float_gpu(
    float * d_dst,
    unsigned char * d_Src,
    int numElements
);
extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    float * c_Kernel,
    int KERNEL_RADIUS
);
extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    float * c_Kernel,
    int KERNEL_RADIUS
);

extern "C" void getImageMoment_GPU(
    float* output, // 1024 float size
    float* input, // input image
    int width, int height,
    float threshold,
    float* d_save // output binary image
);

#endif // CUDA_KERNELS
