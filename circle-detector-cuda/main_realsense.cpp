#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp> 
#include <librealsense2/rs.hpp> 

#include "CircleDetector.h"
#include "CircleDetectorCuda.h"

int main_()
{
    int img_width = 1280;
    int img_height = 720;
    /*int img_width = 1920;
    int img_height = 1080;*/
    int fps = 30;

    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_COLOR, img_width, img_height, RS2_FORMAT_BGR8, fps);

    pipe.start(cfg);

    std::vector<Feature> features;
    DetectionParams det_params;
    det_params.CircleRadius = 50;
    det_params.MassMin      = 2000000;
    det_params.AreaMin      = 100;
    det_params.DensityMin   = 100;
    det_params.WidthMax     = 200;
    det_params.HeightMax    = 200;
    det_params.DiffThreshold = 15;

    //CircleDetector circ_det(det_params, img_width, img_height);
    CircleDetectorCuda circ_det(det_params, img_width, img_height);

    cv::Mat img_gray;

    int elapsed_ms = 0;
    while (true)
    {
        // get a data
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();   

        // create an opencv images
        cv::Mat color(cv::Size(img_width, img_height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(color, img_gray, cv::COLOR_RGB2GRAY);

        // detect circles
        auto t1 = std::chrono::steady_clock::now();
        features = circ_det.Detect(img_gray);
        auto t2 = std::chrono::steady_clock::now();
        elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "circlde detection time: " << elapsed_ms << " ms" << std::endl;

        // show detection
        std::stringstream ss;
        cv::Mat img_show = color.clone();
        circ_det.DrawFeatures(img_show, features, cv::Scalar(255, 0, 255), 1);
        ss<< "detection time " << elapsed_ms << " ms";
        cv::putText(img_show, ss.str(), cv::Point2d(10, 30), 1, 2, cv::Scalar(0, 0, 255));

        cv::imshow("circle detector", img_show);

        if (cv::waitKey(1) >= 0)
            break;
    }

    return 0;
}