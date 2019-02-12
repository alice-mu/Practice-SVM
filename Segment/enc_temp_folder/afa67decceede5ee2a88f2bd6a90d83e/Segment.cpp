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

int SVMpredict(Mat img, SVM* svm);
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

	// insertion sort rectangles from left to right
	for (int i = 1; i < boundRect.size(); i++) {
		Rect currRect = boundRect[i];
		int trav = i - 1;
		while (trav >= 0 && boundRect[trav].tl().x > currRect.tl().x) {
			boundRect[trav + 1] = boundRect[trav];
			trav--;
		}
		boundRect[trav + 1] = currRect;
	}


	string filename = "../Digits/Digits/RBFgridsearch_c1g1.xml";
	Ptr<SVM> svm = Algorithm::load<SVM>(filename);

	vector<int> results;

	// save images into folder
	for (int i = 0; i < boundRect.size(); i++) {
		Rect rect = boundRect[i];
		Range xRange = Range(rect.tl().x, rect.tl().x + rect.width);
		Range yRange = Range(rect.tl().y, rect.tl().y + rect.height);

		Mat oneDigit = bw(yRange, xRange);



		Size size(200, 200);
		resize(oneDigit, oneDigit, size);
		oneDigit.reshape(1, 1);
		string filename = "./Segmented_Digits/";
		filename.append(to_string(i)).append(".jpg");
		imwrite(filename, oneDigit);
		oneDigit = imread(filename);
		int result = SVMpredict(oneDigit, svm);
		results.push_back(result);
	}

	for (int i = 0; i < boundRect.size(); i++) {
		cout << results.at(i);
	}

	waitKey();
	return 0;
}


int SVMpredict(Mat img, SVM* svm) {
	Mat img2;
	img.copyTo(img2);
	Size size(20, 20);
	resize(img2, img2, size);
	img2 = img2.reshape(1, 1);
	img2.convertTo(img2, CV_32F);

	int result = (int)svm->predict(img2);
	return result;
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