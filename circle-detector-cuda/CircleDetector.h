#pragma once

#include <vector>

#include <opencv2\opencv.hpp>

struct DetectionParams
{
	// detector properties
	int DiffThreshold	= 30;
	int CountThreshold	= 0; 
	int CircleRadius	= 31;

	// feature properties
	int MassMin			= 500000;
	int AreaMin			= 100;
	double DensityMin	= 50;
	bool EightConnected = false;

	int WidthMax		= 100;
	int HeightMax		= 100;
	int XMin			= 0;
	int XMax			= 10000;
	int YMin			= 0;
	int YMax			= 10000;
};

struct Feature
{
	int XMin;
	int YMin;
	int XMax;
	int YMax;
	double X;
	double Y;
	double Score;
	double Area;
	double Mass;
	double Density;
	double Orientation;
	double MeanIntesity;
	double StdIntesity;
};


class CircleDetector
{
public:
	CircleDetector(const DetectionParams& params, int img_width, int img_height);
	~CircleDetector();

	virtual std::vector<Feature> CircleDetector::Detect(const cv::Mat& image);	
	void DrawFeatures(cv::Mat& img, const std::vector<Feature>& features, cv::Scalar color, int thickness) const;

	cv::Mat GetResponse() const { return CircleResponse; }
	const DetectionParams& GetParams() const { return Params; }
	

protected:
	DetectionParams Params;

	std::vector<std::pair<int, int>> Coordinates;
	std::vector<Feature> DetectedFeatures;

	cv::Mat Mask;
	cv::Mat CircleResponse;

	uint16_t* CircleResponsePtr;	
	uint8_t* ImagePtr;
	uint8_t* MaskPtr;

	int64_t H, W;
	int64_t MH, MW, MH2, MW2;

	void InitMask();
	void CalcCountThreshold();
	virtual void CalcResponse(const cv::Mat & image);
	virtual void DetectCircleAt(int x, int y);
	void GetCenters(std::vector<Feature>& features);
	void GetCenters(uint16_t* resp_data, std::vector<Feature>& tracking_points, const DetectionParams& params);

};
