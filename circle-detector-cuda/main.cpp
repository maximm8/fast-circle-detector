#include <stdio.h>
#include <chrono>

#include <opencv2/opencv.hpp> 

#include "CircleDetector.h"
#include "CircleDetectorCuda.h"

int main()
{
    cv::Mat img = cv::imread("data/target2.png");
    //cv::Mat img = cv::imread("data/xbox.jpg");

    cv::Mat img_gray;
    cv::Mat img_show = img.clone();
    cv::cvtColor(img, img_gray, cv::COLOR_RGB2GRAY);

    //set detection parameters
    std::vector<Feature> features;
    DetectionParams det_params;
    det_params.CircleRadius     = 45;
    det_params.MassMin          = 500000;
    det_params.AreaMin          = 150;
    det_params.DensityMin       = 100;

    // cpu
    CircleDetector circ_det(det_params, img.size().width, img.size().height);

    auto t11 = std::chrono::steady_clock::now();
    features = circ_det.Detect(img_gray);
    auto t21 = std::chrono::steady_clock::now();
    std::cout << "CPU elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t21 - t11).count() << " ms" << std::endl;

    circ_det.DrawFeatures(img_show, features, cv::Scalar(0, 255, 255), 3);

    // gpu
    CircleDetectorCuda circ_det_cuda(det_params, img.size().width, img.size().height);

    auto t12 = std::chrono::steady_clock::now();
    features = circ_det_cuda.Detect(img_gray);
    auto t22 = std::chrono::steady_clock::now();
    std::cout << "GPU elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count() << " ms" << std::endl;
    
    circ_det.DrawFeatures(img_show, features, cv::Scalar(255, 0, 255), 1);
    
    cv::imshow("circle detector", img_show);
    cv::waitKey(0);

    //cv::imwrite("output.png", img_show);
}