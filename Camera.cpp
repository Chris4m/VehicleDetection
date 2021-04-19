#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <strstream>

#include "MaskRCNN.h"
#include "ElasMatcher.h"
#include "Camera.h"

using namespace std;
using namespace cv;

Camera::Camera(String path, int imagePos) {
	path_ = path;
	getProjectionMatrix();
	getExteriorParameters();
	float f = -P1.at<float>(0, 0);
	calcQ(P1.at<float>(0, 2), P2.at<float>(0, 2), P1.at<float>(1, 2), P2.at<float>(0, 3) / -f, f);
	getCurrentImages(imagePos);
}

int Camera::getProjectionMatrix() {
	ifstream calibData(path_ + "/outCalib.txt");
	std::string line;
	float valuesP1[12];
	float valuesP2[12];
	float value;
	int i = 0;
	if (calibData.is_open())
	{
		while (getline(calibData, line, ' ')) {
			std::istringstream iss(line);
			while (iss >> value) {
				if (i < 12) {
					valuesP1[i] = value;
				}
				else {
					valuesP2[i-12] = value;
				}
				i++;
			}
		}
	}

	Mat P1 = Mat(3, 4, CV_32F, valuesP1);
	Mat P2 = Mat(3, 4, CV_32F, valuesP2);
		
	//cout << P1 << endl;

	this->P1 = P1.clone();
	this->P2 = P2.clone();

	calibData.close();

	return 1;
}

int  Camera::getExteriorParameters() {
	ifstream extParams(path_ + "/absolute.txt");
	std::string line;
	float valuesX0[3];
	float valuesRPY[3];
	float valuesR[9];
	float value;
	int i = 0;
		
	if (extParams.is_open())
	{
		while (getline(extParams, line, ' ')) {
			std::istringstream iss(line);
			while (iss >> value) {
				if (i < 3) {
					valuesX0[i] = value;
				}
				else if (i < 6) {
					valuesRPY[i-3] = value;
				}
				else {
					valuesR[i-6] = value;
				}
				i++;
			}
		}
	}

	Mat X0 = Mat(3, 1, CV_32F, valuesX0);
	Mat R = Mat(3, 3, CV_32F, valuesR);
		
	//cout << X0 << endl;
	//cout << R << endl;
		
	this->RotationMatrixR = R.clone();
	this->X0 = X0.clone();

	extParams.close();

	return 1;
}

int Camera::getCurrentImages(int pos) {
	ifstream extParams(path_ + "/list.txt");
	std::string line;
	int value;
	vector<int> images;
	int i = 0;
		
	if (extParams.is_open())
	{
		while (getline(extParams, line, ' ')) {
			std::istringstream iss(line);
			while (iss >> value) {
				images.push_back(value);
			}
		}
	}

	String imgNr = to_string(images[pos]);

	while (imgNr.length() < 6) {
		imgNr = "0" + imgNr;
	}

	String rightImgPath = path_ + "/right/" + imgNr + ".png";
	rightImg = imread(rightImgPath);
	String leftImgPath = path_ + "/left/" + imgNr + ".png";
	leftImg = imread(leftImgPath);

	return 1;
}

int Camera::calcQ(float cx, float cx_strich, float cy, float Tx, float f) {
	//cout << f << endl;
	//cout << Tx << endl;
	//cout << cx << endl;
	//cout << cy << endl;
	//cout << cx_strich << endl;

	float valuesQ[16] = { 1.0, 0.0, 0.0, -fabs(cx), 0.0, 1.0, 0.0, -fabs(cy), 0.0 , 0.0 , 0.0 , -fabs(f), 0.0 , 0.0 , -1.0 / Tx, (cx - cx_strich) / Tx };

	Mat Q = Mat(4, 4, CV_32F, valuesQ);

	this->ProjectionMatixQ = Q.clone();

	return 1;
}