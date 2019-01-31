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
#include <filesystem>

using namespace std;
using namespace cv;
using namespace cv::ml;

int testSVM(Mat img);
void DrawRectangles(Mat img, vector<Rect> boundRect);

int main()
{
	Mat img = imread("C:/Users/lenovo/Documents/Visual_Studio_Community/Data/handwriting_crop.jpg");
	Mat bw;
	cvtColor(img, bw, CV_BGR2GRAY);
	threshold(bw, bw, 170, 255, THRESH_BINARY_INV);

	vector<Mat> contours;
	Mat bw2;
	bw.copyTo(bw2);

	findContours(bw2, contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


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

	vector<int> results;
	for (int i = 0; i < boundRect.size(); i++) {
		Rect rect = boundRect[i];
		Range xRange = Range(rect.tl().x, rect.tl().x + rect.width);
		Range yRange = Range(rect.tl().y, rect.tl().y + rect.height);
		Mat oneDigit = bw(xRange, yRange);

		imshow("segmented digit", bw);
		waitKey();

		int result = testSVM(oneDigit);
		results.push_back(result);
	}

	waitKey();
	return 0;
}


int testSVM(Mat img) {
	const string SVMfilename = "../Digits/Digits/RBFgridsearch_c1g1.xml";
	Ptr<SVM> svm = SVM::create();
	svm->clear();
	FileStorage svm_fs(SVMfilename, FileStorage::READ);
	if (svm_fs.isOpened())
	{
		svm = Algorithm::load<SVM>(SVMfilename);
	}

	Mat img2;
	img.copyTo(img2);
	img2 = img2.reshape(1, 1);
	img2.convertTo(img2, CV_32F);
	int response = (int)svm->predict(img2);
	return response;
}


void DrawRectangles(Mat img, vector<Rect> boundRect) {
	RNG rng(12345);
	Mat img2;
	img.copyTo(img2);
	for (int i = 0; i < boundRect.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(img2, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
	}

	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	Size size(1500, 750);
	resize(img2, img2, size);
	imshow("Contours", img2);
	waitKey();
}