#pragma once

#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"

#include "CircleDetector.h"


class CircleDetectorCuda: public CircleDetector
{
public:

	CircleDetectorCuda(const DetectionParams& params, int img_width, int img_height);
	~CircleDetectorCuda();

protected:
	virtual void CalcResponse(const cv::Mat& image);

	unsigned char*	img_dev;
	unsigned char*	mask_dev;
	uint16_t*		resp_dev;

};


