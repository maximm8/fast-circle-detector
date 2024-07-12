
#include "CircleDetectorCuda.h"

//#include <stdio.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void checkCudaError(cudaError_t err, const char* msg) 
{

    if (err != cudaSuccess) 
    {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void DetectCircleAtKernel(const unsigned char* img, const unsigned char* mask, uint16_t* response, int w, int h, int mw, int mh, int diff_threshold, int count_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    int mw2 = mw / 2;
    int mh2 = mh / 2;

    int c = img[y * w + x];
    int count = 0;
    uint16_t resp = 0;

    for (int64_t yy = 0; yy < mh; ++yy) 
    {
        for (int64_t xx = 0; xx < mw; ++xx) 
        {
            if (mask[yy * mw + xx]) 
            {
                int16_t j = y + yy - mw2;
                int16_t i = x + xx - mh2;

                if (i>-1 && j>-1 && i < w && j < h)
                {
                    int v = img[j * w + i];

                    int d = abs(c - v);
                    //int d = (v - c);
                    if ( d >= diff_threshold)
                    {
                        count += 1;
                        resp += d;
                    }
                }
            }
        }
    }

    if (count >= count_threshold) 
        response[y * w + x] = resp;    
    else
        response[y * w + x] = 0;
}

CircleDetectorCuda::CircleDetectorCuda(const DetectionParams& params, int img_width, int img_height): CircleDetector(params, img_width, img_height)
{    
    checkCudaError(cudaMalloc((void**)&img_dev,  W * H * sizeof(unsigned char)), "cudaMalloc img_dev");
    checkCudaError(cudaMalloc((void**)&mask_dev, MW * MH * sizeof(unsigned char)), "cudaMalloc mask_dev");
    checkCudaError(cudaMalloc((void**)&resp_dev, W * H * sizeof(uint16_t)), "cudaMalloc resp_dev");

    checkCudaError(cudaMemcpy(mask_dev, Mask.data, MW * MH * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy mask_dew");
}


CircleDetectorCuda::~CircleDetectorCuda()
{
    cudaFree(img_dev);
    cudaFree(mask_dev);
    cudaFree(resp_dev);
}

void CircleDetectorCuda::CalcResponse(const cv::Mat& image)
{
    ImagePtr = image.data;

    // copy data
    checkCudaError(cudaMemcpy(img_dev, image.data, W * H * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy dev_input");

    dim3 blockDims(16, 16);
    dim3 gridDims((W + blockDims.x - 1) / blockDims.x, (H + blockDims.y - 1) / blockDims.y);

    // execute detection
    DetectCircleAtKernel<<< gridDims, blockDims >>>(img_dev, mask_dev, resp_dev, W, H, MW, MH, Params.DiffThreshold, Params.CountThreshold);

    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    //copy to host
    checkCudaError(cudaMemcpy(CircleResponse.data, resp_dev, W * H * sizeof(uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy dev_output");
}