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
	Mat img = imread("C:/Users/lenovo/Documents/Visual_Studio_Community/Data/handwriting_crop.jpg");
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	threshold(gray, gray, 170, 255, THRESH_BINARY_INV);

	vector<Mat> contours;
	Mat gray2;
	gray.copyTo(gray2);

	findContours(gray2, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		boundRect[i] = boundingRect(contours.at(i));
	}

	// remove small contours
	// set minimum size as 1/10000th of the image size
	int minSize = img.rows * img.cols / 10000;
	vector<Rect>::iterator currRect = boundRect.begin();
	while (currRect != boundRect.end()) {
		if (currRect->width * currRect->height < minSize)
			currRect = boundRect.erase(currRect);
		else
			currRect++;
	}


	RNG rng(12345);


	Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
	Mat img2;
	img.copyTo(img2);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(img2, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
	}

	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	Size size(1500, 750);
	resize(img2, img2, size);
	imshow("Contours", img2);

	waitKey();
	return 0;
}