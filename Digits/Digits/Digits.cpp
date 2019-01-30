#include "pch.h"
#include <stdio.h>  
#include <opencv2/opencv.hpp>  
#include <opencv/cv.h>  
#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <io.h>
#include <array>

#define TRAIN

using namespace std;
using namespace cv;
using namespace cv::ml;


#ifdef TRAIN

// Train

void getFiles(string path, vector<string>& files);
void getImages(Mat& trainingImages, vector<int>& trainingLabels, int number);
string trainSVM(Mat trainingImages, vector<int> trainingLabels, double c, double gamma, double degree = 0, double coef0 = 0, double nu = 0, double p = 0);
void getTestFiles(string path, vector<string>& files);
int testSVM(int& mistakes, int* result, string filename);
void summary(int* result, string filename, int num_files, int mistakes);

int main()
{
	//get data
	Mat trainingImages;
	vector<int> trainingLabels;

	for (int i = 0; i < 10; i++) {
		getImages(trainingImages, trainingLabels, i);
	}

	const int CARRAYSIZE = static_cast<int>(2);
	const int GAMMAARRAYSIZE = static_cast<int>(2);

	//set grid search
	double cValues[CARRAYSIZE] = { 1.0, 10.0 };
	double gammaValues[GAMMAARRAYSIZE] = { 1.0, 10.0 };

	for (int i = 0; i < CARRAYSIZE; i++) {
		for (int j = 0; j < GAMMAARRAYSIZE; j++) {
			string filename = trainSVM(trainingImages, trainingLabels, cValues[i], gammaValues[j]);

			int result[10];
			for (int i = 0; i < 10; i++) {
				result[i] = 0;
			}
			int mistakes = 0;

			int num_tested = testSVM(mistakes, result, filename);

			summary(result, filename, num_tested, mistakes);
		}
	}

	waitKey();
	return 0;
}

//Mat deskew(Mat& image)
//{
//	int SZ = 20;
//	float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;
//
//	Mat image2;
//	image.copyTo(image2);
//	image2.convertTo(image2, CV_32F);
//	Moments m = moments(image2);
//
//	if (m.mu02 < 0.01 && m.mu02 > -0.01) {
//		return image2.clone();
//	}
//
//	float skew = m.mu11 / m.mu02;
//	Mat warpMat = (Mat_<float>(2, 3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
//	Mat imgOut = Mat::zeros(image2.rows, image2.cols, image2.type());
//	warpAffine(image2, imgOut, warpMat, imgOut.size(), affineFlags);
//
//	return imgOut;
//}

void getFiles(string path, vector<string>& files)
{
	string p;
	string j;

	for (int i = 0; i < 400; i++) {
		j = std::to_string(i);
		p = path;
		files.push_back(p.append("/").append(j).append(".jpg"));
	}
}


void getImages(Mat& trainingImages, vector<int>& trainingLabels, int number)
{
	string filePath = "C:/Users/602151/Documents/Visual_Studio_Community/data/digits/";
	filePath.append(std::to_string(number));
	vector<string> files;
	getFiles(filePath, files);
	int files_size = files.size();
	for (int i = 0; i < files_size; i++)
	{
		Mat  SrcImage = imread(files[i].c_str());
		SrcImage = SrcImage.reshape(1, 1);
		//Mat deskewed = deskew(SrcImage);
		//deskewed = deskewed.reshape(1, 1);
		//imshow("deskewed", deskewed);
		//waitKey();
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(number);
	}
}

void summary(int* result, string filename, int num_files, int mistakes) {
	cout << "testing summary for " << filename << endl;
	cout << "total images:" << num_files << endl;
	cout << "total mistakes: " << mistakes << endl << "accuracy: " << 100.0 - ((float)mistakes / (float)num_files) * 100 << endl;
	for (int i = 0; i < 10; i++) {
		cout << "there are " << result[i] << " images that are a " << i << endl;
	}
	cout << endl << endl << endl;
}


// return filename of file containing trained SVM
string trainSVM(Mat trainingImages, vector<int> trainingLabels, double c, double gamma, double degree, double coef0, double nu, double p) {
	string filename = "RBFgridsearch_";
	filename.append("c").append(to_string(int(c))).append("g").append(to_string(int(gamma))).append(".xml");

	cout << "training " << filename << "..." << endl;
	cout << "c:" << c << "   gamma:" << gamma << "   degree:" << degree << "   coef0:" << coef0 << "   nu:" << nu << "   p:" << p << endl;

	string filename2 = filename;
	struct _finddata_t fileinfo;
	int fileFound = _findfirst(filename2.c_str(), &fileinfo);
	if (fileFound != -1) {
		cout << "found file " << filename << ", skipping training" << endl << endl;
	}
	else {
		Mat trainingImages2;
		trainingImages.copyTo(trainingImages2);
		Mat trainingLabels2;
		Mat(trainingLabels).copyTo(trainingLabels2);

		trainingImages2.convertTo(trainingImages2, CV_32F);
		Ptr<TrainData> trainingData = TrainData::create(trainingImages2, ROW_SAMPLE, trainingLabels2);

		//intialize SVM to start training
		Ptr<SVM> svm = SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::RBF);
		svm->setDegree(degree);
		svm->setGamma(gamma);
		svm->setCoef0(coef0);
		svm->setC(c);
		svm->setNu(nu);
		svm->setP(p);
		svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01));

		//train
		svm->trainAuto(trainingData);
		//save model
		svm->save(filename);

		cout << "finished training " << filename << endl << endl;
	}
	return filename;
};



void getTestFiles(string path, vector<string>& files) {
	string folder;
	string file;

	for (int j = 0; j < 10; j++) {
		folder = path;
		folder.append("/").append(to_string(j));

		for (int i = 400; i < 500; i++) {
			file = folder;
			files.push_back(file.append("/").append(to_string(i)).append(".jpg"));
		}
	}
}


// returns number of files tested on
int testSVM(int& mistakes, int* result, string filename) {
	cout << "testing..." << endl;
	const char * filePath = "C:/Users/602151/Documents/Visual_Studio_Community/data/digits";
	vector<string> files;
	getTestFiles(filePath, files);
	int num_files = files.size();
	Ptr<SVM> svm = SVM::create();
	svm->clear();

	FileStorage svm_fs(filename, FileStorage::READ);
	if (svm_fs.isOpened())
	{
		svm = Algorithm::load<SVM>(filename);
	}

	for (int i = 0; i < num_files; i++)
	{
		Mat inMat = imread(files[i].c_str());
		Mat p = inMat.reshape(1, 1);
		p.convertTo(p, CV_32F);
		int response = (int)svm->predict(p);

		result[response]++;

		if (i / 100 != response) {
			//cout << "predicted: " << response << "   actual: " << (i / 100);
			//cout << "   image: " << 400 + i - (i / 100) * 100 << ".jpg" << endl;
			mistakes++;
		}
	}
	return num_files;
}

#endif



#ifdef TEST
//Test

void getFiles(string path, vector<string>& files);

int main()
{
	int result[10];
	for (int i = 0; i < 10; i++) {
		result[i] = 0;
	}

	const char * filePath = "C:/Users/lenovo/Documents/Visual_Studio_Community/data/digits";
	vector<string> files;
	getFiles(filePath, files);
	int num_files = files.size();
	// cout << num_files << endl;
	Ptr<SVM> svm = SVM::create();
	svm->clear();
	string modelpath = "digits_trained.xml";
	FileStorage svm_fs(modelpath, FileStorage::READ);
	if (svm_fs.isOpened())
	{
		svm = Algorithm::load<SVM>(modelpath);
	}

	//int a = 150;
	//cout << (a / 100) * 100 << endl;

	int mistakes = 0;

	for (int i = 0; i < num_files; i++)
	{
		Mat inMat = imread(files[i].c_str());
		Mat p = inMat.reshape(1, 1);
		p.convertTo(p, CV_32F);
		int response = (int)svm->predict(p);

		result[response]++;

		if (i / 100 != response) {
			cout << "predicted: " << response << "   actual: " << (i / 100);
			cout << "   image: " << 400 + i - (i / 100) * 100 << ".jpg" << endl;
			mistakes++;
		}
	}
	cout << endl << "summary" << endl;
	cout << "total images:" << num_files << endl;
	cout << "total mistakes: " << mistakes << endl << "accuracy: " << 100.0 - ((float)mistakes / (float)num_files) * 100 << endl;
	for (int i = 0; i < 10; i++) {
		cout << "there are " << result[i] << " images that are a " << i << endl;
	}

	getchar();
	return  0;
}
void getFiles(string path, vector<string>& files)
{
	string folder;
	string file;

	for (int j = 0; j < 10; j++) {
		folder = path;
		folder.append("/").append(to_string(j));

		for (int i = 400; i < 500; i++) {
			file = folder;
			files.push_back(file.append("/").append(to_string(i)).append(".jpg"));
		}
	}
}





#endif



#ifdef DATA
// Data

int main()
{
	char ad[128] = { 0 };
	int  filename = 0, filenum = 0;
	Mat img = imread("C:/Users/lenovo/Documents/Visual_Studio_Community/data/digits.png");
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	int b = 20;
	int m = gray.rows / b;   //original image is 1000*2000
	int n = gray.cols / b;   //divide into 5000 20*20 squares

	//imshow("gray digits", gray);
	//waitKey();
	//imwrite("C:/Users/lenovo/Documents/Visual_Studio_Community/data/test.jpg", gray);
	//return 0;


	////create the subdirectories for digits 0 - 10
	//for (int i = 0; i < 10; i++) {
	//	const char* path = "C:/User/lenovo/Documents/Visual_Studio_Community/data/digits/" + i;
	//	int hey = 23;
	//	_mkdir(path);
	//}

	for (int i = 0; i < m; i++)
	{
		int offsetRow = i * b;
		if (i % 5 == 0 && i != 0)
		{
			filename++;
			filenum = 0;
		}
		for (int j = 0; j < n; j++)
		{

			int offsetCol = j * b;

			sprintf_s(ad, "C:/Users/lenovo/Documents/Visual_Studio_Community/data/digits/%d/%d.jpg", filename, filenum++);

			Mat tmp;
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			imwrite(ad, tmp);
		}
	}
	return 0;
}

#endif



#ifdef RECOGNIZE
int main()
{
	//Mat img = imread("C:/Users/lenovo/Documents/Visual_Studio_Community/Data/handwriting_crop.jpg");
	Mat img = imread("C:/Users/602151/Documents/Visual_Studio_Community/Data/dot2.jpg");

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	threshold(gray, gray, 170, 255, THRESH_BINARY_INV);
	//Canny(img, img, 30, 128, 3, false);
   //adaptiveThreshold(img, img, 180, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 0);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat gray2;
	gray.copyTo(gray2);
	//imwrite("test.jpg", gray2);

	findContours(gray2, contours, hierarchy, RETR_LIST, CV_CHAIN_APPROX_NONE);

	cout << "contour size: " << contours.size() << endl;

	//vector<vector<Point> >::iterator itrContour = contours.begin();

	////// remove smaller contoursifdef
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
		approxPolyDP(contours.at(i), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours.at(i));

		//vector<Point> curContour = contours.at(i);
		//
		//for (int j = 0; j < curContour.size(); j++)
		//	cout << curContour[j].x << "  " << curContour[j].y << endl;

		cout << contours[i] << endl;
	}

	//waitKey();
	//return 0;

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


//#include "pch.h"
//#include <stdio.h>  
//#include <time.h>  
//#include <opencv2/opencv.hpp>  
//#include <opencv/cv.h>  
//#include <iostream> 
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp>  
//#include <opencv2/ml/ml.hpp>  
//#include <io.h>                                                 
///*
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/video.hpp>
//#include <iostream>
//#include <cmath>
//#include <algorithm>
//#include <vector>
//#include <math.h>
//#include <stdio.h>
//*/
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	Mat img = imread("C:/Users/602151/Documents/Visual_Studio_Community/Data/dot2.jpg", IMREAD_COLOR);
//
//	Mat gray;
//	cvtColor(img, gray, CV_BGR2GRAY);
//
//	threshold(gray, gray, 170, 255, THRESH_BINARY_INV);
//
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	Mat gray2;
//	gray.copyTo(gray2);
//	//imwrite("test.jpg", gray2);
//
//	findContours(gray2, contours, hierarchy, RETR_LIST, CV_CHAIN_APPROX_NONE);
//
//	cout << "contour size: " << contours.size() << endl;
//
//	vector<vector<Point> > contours_poly(2);
//	vector<Rect> boundRect(2);
//	for (int i = 0; i < contours.size(); i++)
//	{
//		approxPolyDP(contours.at(i), contours_poly[i], 3, true);
//
//		boundRect[i] = boundingRect(contours.at(i));
//
//		cout << boundRect[i] << endl;
//		cout << contours[i] << endl;
//	}
//
//
//
//	Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
//	for (int i = 0; i < contours.size(); i++)
//	{
//		drawContours(img, contours, i, Scalar(0, 0, 255), 3, 8, vector<Vec4i>(), 0, Point());
//
//		rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 0, 255), 2, 8, 0);
//	}
//
//	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//	imshow("Contours", img);
//
//	waitKey();
//	return 0;
//}
//
//
//
//

#endif