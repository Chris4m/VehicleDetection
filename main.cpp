#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <strstream>

#include "MaskRCNN.h"
#include "ElasMatcher.h"
#include "../Include/Camera.h"

using namespace std;
using namespace cv;

int IMAGE_POSITION = 0;
int COUNT_SAMPLE = 29;
int DIST_TRESH = 3;

vector<pair<Mat, Vec3b>> eliminateOutlier(vector<pair<Mat, Vec3b>> car) {
	int matSize = car.size();
	int maxCount = 0;
	int bestPoint = 0;
	vector<pair<Mat, Vec3b>> circlePoints;
	vector<Point2f> points2D;
	for (int i = 0; i < COUNT_SAMPLE; i++) {
		int randPos = rand() % matSize;
		Mat compareCoord = car[randPos].first;
		int count = 0;
		for (int j = 0; j < car.size(); j++) {
			float dist = sqrt(pow(car[j].first.at<float>(0, 0) - compareCoord.at<float>(0, 0), 2) + pow(car[j].first.at<float>(0, 1) - compareCoord.at<float>(0, 1), 2));
			if (dist < DIST_TRESH) {
				count++;
			}
		}
		if (count > maxCount) {
			maxCount = count;
			bestPoint = randPos;
		}
	}

	Mat bestCoord = car[bestPoint].first;
	for (int i = 0; i < car.size(); i++) {
		float dist = sqrt(pow(car[i].first.at<float>(0, 0) - bestCoord.at<float>(0, 0), 2) + pow(car[i].first.at<float>(0, 1) - bestCoord.at<float>(0, 1), 2));
		if (dist < DIST_TRESH) {
			circlePoints.push_back(car[i]);
			Point2f point = (car[i].first.at<float>(0, 0), car[i].first.at<float>(0, 1));
			points2D.push_back(point);
		}
	}

	return circlePoints;
}

pair<vector<pair<Mat, Vec3b>>, Point3f> computeCenter(vector<pair<Mat, Vec3b>> points) {
	float sumX = 0.0;
	float sumY = 0.0;
	float sumZ = 0.0;
	for (int k = 0; k < points.size(); k++) {
		sumX += points[k].first.at<float>(0, 0);
		sumY += points[k].first.at<float>(1, 0);
		sumZ += points[k].first.at<float>(2, 0);
	}

	Point3f center(sumX / points.size(), sumY / points.size(), sumZ / points.size());

	pair<vector<pair<Mat, Vec3b>>, Point3f> car(points, center);
	return car;
}


int main(int argc, char *argv[])
{	
	// EXAMPLE CODE

	// initialise the mask RCNN
	MaskRCNN maskRCNN;
	maskRCNN.init("parameterFiles\\MaskRCNNDefinitionFile.txt");

	// initialise the dense matcher
	ElasMatcher matcher;
	
	

	// Initializing Cameras 
	Camera cam1 = Camera("Camera1_AVT", IMAGE_POSITION);
	Camera cam2 = Camera("Camera2_PG", IMAGE_POSITION);
	Camera cam3 = Camera("Camera3_PG_Velodyne", IMAGE_POSITION);

	//Alle Kameras in einem Vector speichern
	vector<Camera> cams;
	cams.push_back(cam1);
	cams.push_back(cam2);
	cams.push_back(cam3);

	/*vector mit allen detektierten Autos (Punktwolke und Zentrum*/
	vector<pair<vector<pair<Mat, Vec3b>>, Point3f>> detectedCars;

	//for Schleife über alle 3 Kameras
	for (int i = 0; i < 3; i++) {
		//int i = 0;

		// detect vehicles
		Mat imgSeg;
		int nVehicles;
		maskRCNN.detectCars(cams[i].leftImg, imgSeg, nVehicles, 0.7, 0.3);
		cout << "Image" << i << ": " << nVehicles << " Autos" << endl;

		// dense matching
		Mat disp_lr, disp_rl;
		matcher.match(cams[i].leftImg, cams[i].rightImg, disp_lr, disp_rl);
		
		// Zeilen und Spalten des Bildes
		Size s = imgSeg.size();
		int rows = s.height;
		int cols = s.width;

		/*Vektoren vorbereiten: 
		Pro Auto eine Matrix mit seinen Disparitäten und 
		Pro Auto ein Vector mit den Pixelkoordinaten welche zu diesem Auto gehören*/
		vector<Mat> disp_cars;
		//vector<vector<Point2i>> cars2D;
		vector<vector<pair<Point2i,Vec3b>>> cars2D;
		cars2D.clear();

		for (int j = 0; j < nVehicles; j++) {
			Mat car = (Mat_<float>(rows, cols));
			disp_cars.push_back(car);
			vector<pair<Point2i, Vec3b>> car2D;
			cars2D.push_back(car2D);
		}


		/*Vektoren befüllen*/
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < rows; k++) {

				int8_t carID = imgSeg.at<int8_t>(k, j);
				if (carID != 0) {

					int pos = (int)carID - 1;
					Point2i point(k, j);
					Vec3b intensity = cams[i].leftImg.at<Vec3b>(k, j);
					//cout << to_string(intensity.val[0]) << to_string(intensity.val[1]) << to_string(intensity.val[2]) << endl;
					pair<Point2i, Vec3b> carSegment (point, intensity);
					cars2D[pos].push_back(carSegment);

					float disp = disp_lr.at<float>(k, j);
					if (disp >= 50.0) {
						disp_cars[pos].at<float>(k, j) = disp;
					} else {
						disp_cars[pos].at<float>(k, j) = 0;
					}
				}	
			}
		}


		// Visualization ________________________________
		//namedWindow("image " + to_string(i), WINDOW_NORMAL);
		//imshow("image " + to_string(i), cams[i].leftImg);
		
		//namedWindow("disparity " + to_string(i), WINDOW_NORMAL);
		//disp_lr.convertTo(disp_lr, CV_8UC1);
		//imshow("disparity " + to_string(i), disp_lr);
		
		//namedWindow("segmentationMask " + to_string(i), WINDOW_NORMAL);
		//imgSeg.convertTo(imgSeg, CV_8UC1, 255.0 / nVehicles);
		//imshow("segmentationMask " + to_string(i), imgSeg);

		for (int j = 0; j < nVehicles; j++) {
			//namedWindow("cam " + to_string(i) + " car " + to_string(j), WINDOW_NORMAL);
			//disp_cars[j].convertTo(disp_cars[j], CV_8UC1);
			//imshow("cam " + to_string(i) + " car " + to_string(j), disp_cars[j]);

			vector<Point3f> car3D;
			car3D.clear();
			vector<pair<Mat, Vec3b>> car3dGloal;
			car3dGloal.clear();
			Mat img3D;
			reprojectImageTo3D(disp_cars[j], img3D, cams[i].ProjectionMatixQ);
			for (int k = cars2D[j].size() - 1; k >= 0; k--) {
				Point3d point = img3D.at<Vec3f>(cars2D[j][k].first.x, cars2D[j][k].first.y);
				/*Wenn keine 3D koordinate zu der Pixelkoordinate gehört, wird diese aus dem Vektor gelöscht*/
				if (isinf(abs(point.x)) || isinf(abs(point.y)) || isinf(abs(point.z))) {
					cars2D[j].erase(cars2D[j].begin() + k);
				}
			}
			for (int k = 0; k < cars2D[j].size(); k++) {
				Point3d point = img3D.at<Vec3f>(cars2D[j][k].first.x, cars2D[j][k].first.y);
				Mat pointAsMat (Mat_<float>(3, 1));
				pointAsMat.at<float>(0, 0) = point.x;
				pointAsMat.at<float>(1, 0) = point.y;
				pointAsMat.at<float>(2, 0) = point.z;
				car3D.push_back(point);
				//cout << point.x << " " << point.y << " " << point.z << endl;
				Mat coordGlob = cams[i].RotationMatrixR * pointAsMat + cams[i].X0;
				pair<Mat, Vec3b> coordGlobWithColor(coordGlob, cars2D[j][k].second);
				car3dGloal.push_back(coordGlobWithColor);
			}

			if (car3dGloal.size() > 0) {
				vector<pair<Mat, Vec3b>> eliminatedOutlier = eliminateOutlier(car3dGloal);

				pair<vector<pair<Mat, Vec3b>>, Point3f> car = computeCenter(eliminatedOutlier);
				
				detectedCars.push_back(car);
				//cams[i].global3Dcars.push_back(car);
			}
		}

	}

	/*gleiche Autos erkennen und zusammenfügen*/
	bool changesDetected = true;
	while (changesDetected) { // Wird solange wiederholt, bis keine Autos mehr zusammengefügt worden sind
		changesDetected = false;
		for (int i = 0; i < detectedCars.size(); i++) { // Schleife über alle Autos
			Point3f centerCar1 = detectedCars[i].second;
			for (int j = i + 1; j < detectedCars.size(); j++) { // Vergleich mit allen anderen Autos 
				Point3f centerCar2 = detectedCars[j].second;
				float dist = sqrt(pow(centerCar1.x - centerCar2.x, 2) + pow(centerCar1.y - centerCar2.y, 2));
				if (dist < DIST_TRESH) { // Auto i und j sind Schwerpunkte nahe beieinander
					cout << "Auto " << i << " und " << j << " zusammengebaut" << endl;
					changesDetected = true;
					for (int k = 0; k < detectedCars[j].first.size(); k++) { // Alle Punkte aus j werten zu i hinzugefügt
						detectedCars[i].first.push_back(detectedCars[j].first[k]);
					}
					detectedCars[i] = computeCenter(detectedCars[i].first); // Neues Zentrum der Punkte wird berechnet
					detectedCars.erase(detectedCars.begin() + j); // Auto j wird aus Liste gelöscht
					break;
				}
			}
			if (changesDetected) {
				break;
			}
		}
	}


	
	for (int i = 0; i < detectedCars.size(); i++) {
		
		/* Speichern der Daten in txt-files*/
		ofstream outputfile;
		outputfile.open("output/car" + to_string(i) + ".txt");
		for (int k = 0; k < detectedCars[i].first.size(); k++) {
			Mat point = detectedCars[i].first[k].first;
			Vec3b color = detectedCars[i].first[k].second;
			outputfile << point.at<float>(0, 0) << " " << point.at<float>(1, 0) << " " << point.at<float>(2, 0) << " " << to_string(color.val[0]) << " " << to_string(color.val[1]) << " " << to_string(color.val[2]) << endl;
		}
		outputfile.close();

	}
	

	waitKey(0);
	return EXIT_SUCCESS;
}