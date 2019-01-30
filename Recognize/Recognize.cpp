#include "pch.h"
#include <stdio.h>  
#include <time.h>  
#include <opencv2/opencv.hpp>  
#include <opencv/cv.h>  
#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <io.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	//Mat img = imread("C:/Users/lenovo/Documents/Visual_Studio_Community/Data/handwriting_crop.jpg");
	Mat img = imread("C:/Users/lenovo/Documents/Visual_Studio_Community/Data/dot3.jpg");
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	threshold(gray, gray, 170, 255, THRESH_BINARY_INV);
	//Canny(img, img, 30, 128, 3, false);
	//adaptiveThreshold(img, img, 180, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 0);

	vector<Mat> contours;
	Mat gray2;
	gray.copyTo(gray2);

	findContours(gray2, contours, RETR_LIST, CV_CHAIN_APPROX_NONE);

	cout << "contour size: " << contours.size() << endl;

	//vector<vector<Point> >::iterator itrContour = contours.begin();

	////// remove smaller contours
	//while (itrContour != contours.end())
	//{
	//	Rect rectBound = boundingRect(*itrContour);
	//	cout << rectBound << endl;
	////	rectBound.y += rowCount;

	////	if (itrContour->size() < iMinContour || rectBound.height < mBandSize - 2 || rectBound.width < 20 || rectBound.width > 200)
	////	{
	////		itrContour = contours.erase(itrContour);
	////	}
	////	else
	////	{
	////		curBandContours.push_back(rectBound);
	//		++itrContour;
	////	}
	//}





	vector<vector<Point> > contours_poly(2);
	vector<Rect> boundRect(2);
	for (int i = 0; i < contours.size(); i++)
	{
		//approxPolyDP(contours.at(i), contours_poly[i], 3, true);
		//boundRect[i] = boundingRect(contours.at(i));

		//vector<Point> curContour = contours.at(i);
		//
		//for (int j = 0; j < curContour.size(); j++)
		//	cout << curContour[j].x << "  " << curContour[j].y << endl;

		//cout << contours[i] << endl;
	}

	RNG rng(12345);


	Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(img, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		//cout << boundRect[i].tl() << endl;
		//cout << boundRect[i].br() << endl;

		rectangle(img, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
	}

	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", img);

	//Size size(1000, 1000);
	//resize(gray, gray, size);
	//imshow("test", gray);

	waitKey();
	return 0;
}