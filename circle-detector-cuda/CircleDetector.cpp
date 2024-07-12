#include "CircleDetector.h"

CircleDetector::CircleDetector(const DetectionParams& params, int img_width, int img_height): Params(params)
{
	W = img_width;
	H = img_height;

	InitMask();
	CalcCountThreshold();

	CircleResponse = cv::Mat::zeros(H, W, CV_16UC1);
	CircleResponsePtr = (uint16_t*)CircleResponse.data;
}

CircleDetector::~CircleDetector()
{
}

void CircleDetector::DrawFeatures(cv::Mat& img, const std::vector<Feature>& features, cv::Scalar color, int thickness) const
{
	for (std::vector<Feature>::const_iterator it = features.cbegin(); it != features.cend(); ++it)
	{
		cv::Point2f pp = cv::Point2f(it->XMin, it->YMin);
		cv::Point2f rect_size(it->XMax-it->XMin, it->YMax-it->YMin);
		cv::Point2f pp2 = cv::Point2f(it->X - 1, it->Y - 1);
		cv::Point2f rect_size2(3, 3);

		cv::rectangle(img, pp, pp + rect_size, color,  thickness);
		cv::rectangle(img, pp2, pp2 + rect_size2, color, -1);
	}
}

void CircleDetector::InitMask()
{
	int circle_diameter = Params.CircleRadius * 2 + 1;

	Mask = cv::Mat::zeros(circle_diameter, circle_diameter, CV_8UC1);

	cv::circle(Mask, cv::Point(Params.CircleRadius, Params.CircleRadius), Params.CircleRadius + 1, cv::Scalar(1, 1, 1), 1);
	Mask.at<uint8_t>(Params.CircleRadius, 0) = 1;
	Mask.at<uint8_t>(0, Params.CircleRadius) = 1;
	Mask.at<uint8_t>(Params.CircleRadius, Mask.size().width - 1) = 1;
	Mask.at<uint8_t>(Mask.size().height - 1, Params.CircleRadius) = 1;
	//}

	MH = Mask.size().height;
	MW = Mask.size().width;
	MH2 = (MH - 1) / 2;
	MW2 = (MW - 1) / 2;

	MaskPtr = Mask.data;
}

void CircleDetector::CalcCountThreshold()
{
	if (Params.CountThreshold == 0)
	{
		int count = 0;
		for (int i = 0; i < Mask.size().area(); ++i)
			if (MaskPtr[i] == 1)
				count++;

		Params.CountThreshold = count * 0.8;
	}
}

std::vector<Feature> CircleDetector::Detect(const cv::Mat& image)
{
	DetectedFeatures.clear();

	CalcResponse(image);
	GetCenters(DetectedFeatures);

	return DetectedFeatures;
}

void CircleDetector::CalcResponse(const cv::Mat& image)
{	
	ImagePtr = image.data;

	int y_lim = MIN(H, Params.YMax);
	int x_lim = MIN(W, Params.XMax);

#pragma omp parallel for
	for (int64_t y = Params.YMin + MH2; y < y_lim - MH2; ++y)
	{
#pragma omp parallel for
		for (int64_t x = Params.XMin + MW2; x < x_lim - MW2; ++x)
		{
			DetectCircleAt(x, y);
		}
	}
}

void CircleDetector::DetectCircleAt(int x, int y)
{
	int c = ImagePtr[y * W + x];
	int count = 0;
	float resp = 0;
	

	for (int64_t yy = 0; yy < MH; ++yy)
	{
		for (int64_t xx = 0; xx < MW; ++xx)
		{
			if (MaskPtr[yy * MW + xx])
			{
				//count2++;
				int64_t j = y + yy - MW2;
				int64_t i = x + xx - MH2;
				int v = ImagePtr[j * W + i];

				int d = abs(c - v);
				//int d = (v - c);
				if (d >= Params.DiffThreshold)
				{
					count += 1;
					resp += d;
				}
			}
		}
	}

	if (count >= Params.CountThreshold)
		CircleResponsePtr[y * W + x] = resp;
	else
		CircleResponsePtr[y * W + x] = 0;

}

void CircleDetector::GetCenters(std::vector<Feature>& features)
{
	GetCenters(CircleResponsePtr, features, Params);
}


void CircleDetector::GetCenters(uint16_t* resp_data, std::vector<Feature>& features, const DetectionParams& params)
{
	float threshold = 0;
	cv::Mat visited = cv::Mat(H, W, CV_8UC1);
	visited.setTo(0);
	uint8_t* visited_data = (uint8_t*)(visited.data);

	for (size_t y = 0; y < H; ++y)
	{
		for (size_t x = 0; x < W; x++)//+= 4)
		{
			size_t offset = x + y * W;


			if (resp_data[offset] > threshold)
			{
				double xsum = 0, ysum = 0, m = 0, a = 0, mean_int = 0;
				int x_min = W, x_max = 0, y_min = H, y_max = 0;

				std::list<cv::Point2d> to_visit;
				to_visit.push_back(cv::Point2d(x, y));

				while (to_visit.size() > 0)
				{
					cv::Point2d point = to_visit.front();
					to_visit.pop_front();

					if (point.x > x_max)
						x_max = static_cast<int>(point.x);
					if (point.y > y_max)
						y_max = static_cast<int>(point.y);
					if (point.x < x_min)
						x_min = static_cast<int>(point.x);
					if (point.y < y_min)
						y_min = static_cast<int>(point.y);

					size_t index = point.x + point.y * W;

					if (visited_data[index] != 0)
						continue;

					visited_data[index] = 1;

					xsum += point.x * resp_data[index];
					ysum += point.y * resp_data[index];
					mean_int += ImagePtr[index];
					m += resp_data[index];
					a++;

					if (point.x + 1 < W)
					{
						size_t offset1 = (point.x + 1) + point.y * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x + 1, point.y));
					}

					if (point.x - 1 > -1)
					{
						size_t offset1 = (point.x - 1) + point.y * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x - 1, point.y));
					}

					if (point.y + 1 < H)
					{
						size_t offset1 = (point.x) + (point.y + 1) * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x, point.y + 1));
					}

					if (point.y - 1 > -1)
					{
						size_t offset1 = (point.x) + (point.y - 1) * W;

						if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
							to_visit.push_back(cv::Point2d(point.x, point.y - 1));
					}

					if (params.EightConnected)
					{
						if (point.x + 1 < W && point.y + 1 < H)
						{
							size_t offset1 = (point.x + 1) + (point.y + 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x + 1, point.y + 1));
						}

						if (point.x + 1 < W && point.y - 1 > -1)
						{
							size_t offset1 = (point.x + 1) + (point.y - 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x + 1, point.y - 1));
						}

						if (point.x - 1 > -1 && point.y + 1 < H)
						{
							size_t offset1 = (point.x - 1) + (point.y + 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x - 1, point.y + 1));
						}

						if (point.x - 1 > -1 && point.y - 1 > -1)
						{
							size_t offset1 = (point.x - 1) + (point.y - 1) * W;

							if (resp_data[offset1] > threshold && visited_data[offset1] == 0)
								to_visit.push_back(cv::Point2d(point.x - 1, point.y - 1));
						}
					}
				}

				double density = m / a;
				if (m > params.MassMin && a > params.AreaMin && density > params.DensityMin)
				{
					if (x_max - x_min < params.WidthMax && y_max - y_min < params.HeightMax)
					{
						double x_out = xsum / m;
						double y_out = ysum / m;
						mean_int /= a;
						double orient = 0;

						if (x_out > params.XMin && x_out < params.XMax && y_out > params.YMin && y_out < params.YMax)
						{
							Feature tp;
							tp.X			= x_out;
							tp.Y			= y_out;
							tp.Area			= a;
							tp.Mass			= m;
							tp.Density		= density;
							tp.Orientation	= orient;
							tp.XMin			= x_min;
							tp.YMin			= y_min;
							tp.XMax			= x_max;
							tp.YMax			= y_max;
							tp.MeanIntesity = mean_int;
							
							features.push_back(tp);
						}
					}
				}
			}
		}
	}
}
