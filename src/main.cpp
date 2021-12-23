#include <iostream>
#include "Utils.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

vector<int> getContoursBits(Mat image, vector<Point2f> cnt, int bits) {
	assert(bits > 0 && round(sqrt(bits)) == sqrt(bits));
	int pixelLen = sqrt(bits);

	// NOTE: we assume contour points are clockwise starting from top-left
	vector<Point2f> corners = { Point2f(0, 0), Point2f(bits, 0), Point2f(bits, bits), Point2f(0, bits) };
	Mat M = getPerspectiveTransform(cnt, corners);
	Mat warpped;
	warpPerspective(image, warpped, M, Size(bits, bits));

	Mat binary;
	threshold(warpped, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Mat c;
	cvtColor(binary, c, COLOR_GRAY2BGR);
	Utils::drawGrid(c, pixelLen, pixelLen);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(binary, binary, element);

	Mat c2;
	cvtColor(binary, c2, COLOR_GRAY2BGR);
	Utils::drawGrid(c2, pixelLen, pixelLen);

	vector<int> res;
	for (int r = 0; r < pixelLen; ++r) {
		for (int c = 0; c < pixelLen; ++c) {
			int y = r * pixelLen + (pixelLen / 2);
			int x = c * pixelLen + (pixelLen / 2);
			if (binary.at<uchar>(y, x) >= 128)
			{
				res.push_back(1);
			}
			else
			{
				res.push_back(0);
			}
		}
	}

	return res;
}

bool equalSig(vector<int>& sig1, vector<int>& sig2, int allowedMisses = 0)
{
	int misses = 0;
	for (int i = 0; i < sig1.size(); ++i) {
		if (sig1[i] != sig2[i])
			++misses;
	}

	return misses <= allowedMisses;
}

void orderContour(vector<Point2f>& cnt)
{
	float cx = (cnt[0].x + cnt[1].x + cnt[2].x + cnt[3].x) / 4.0f;
	float cy = (cnt[0].y + cnt[1].y + cnt[2].y + cnt[3].y) / 4.0f;

	// IMPORTANT! We assume the contour points are counter-clockwise (as we use EXTERNAL contours in findContours)
	if (cnt[0].x <= cx && cnt[0].y <= cy)
	{
		swap(cnt[1], cnt[3]);
	}
	else
	{
		swap(cnt[0], cnt[1]);
		swap(cnt[2], cnt[3]);
	}
}

vector<vector<Point2f>> findSquares(Mat img) {
	vector<vector<Point2f>> cand;

	Mat thresh;
	adaptiveThreshold(img, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 5);

	thresh = ~thresh;
	vector<vector<Point>> cnts;
	findContours(thresh, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<Point2f> cnt;
	for (int i = 0; i < cnts.size(); ++i)
	{
		approxPolyDP(cnts[i], cnt, 0.05 * arcLength(cnts[i], true), true);
		if (cnt.size() != 4 || contourArea(cnt) < 200 || !isContourConvex(cnt)) {
			continue;
		}

		cornerSubPix(img, cnt, Size(5, 5), Size(-1, -1), TERM_CRIT);

		orderContour(cnt);
		cand.push_back(cnt);
	}

	return cand;
}

vector<ArucoResult> detectArucos(Mat frame, ArucoDict dict, int misses = 0) {
	vector<ArucoResult> res;
	vector<vector<Point2f>> cands = findSquares(frame);

	for (int i = 0; i < cands.size() && res.size() < 3; ++i) {
		vector<Point2f> cnt = cands[i];
		vector<int> sig = getContoursBits(frame, cnt, 36);
		for (int j = 0; j < dict.sigs.size(); ++j) {
			if (equalSig(sig, dict.sigs[j], misses)) {
				ArucoResult ar;
				ar.corners = cnt;
				ar.index = j;
				res.push_back(ar);
				break;
			}
		}
	}

	return res;
}

ArucoDict loadMarkerDictionary(Mat marker, int bits) {
	ArucoDict res;
	int w = marker.cols;
	int h = marker.rows;

	cvtColor(marker, marker, COLOR_BGRA2GRAY);

	vector<Point2f> cnt = { Point2f(0, 0), Point2f(w, 0) , Point2f(w, h) , Point2f(0, h) };
	vector<Point3f> world = { Point3f(0, 0, 0), Point3f(25, 0, 0), Point3f(25, 25, 0), Point3f(0, 25, 0) };

	for (int i = 0; i < 4; ++i) {
		vector<int> sig = getContoursBits(marker, cnt, bits);
		res.sigs.push_back(sig);

		vector<Point3f> w(world);
		res.worldLoc.push_back(w);

		rotate(marker, marker, ROTATE_90_CLOCKWISE);

		world.insert(world.begin(), world[3]);
		world.pop_back();
	}

	return res;
}

void main() 
{
	Mat marker = imread("imgs/marker.png");
	ArucoDict dict = loadMarkerDictionary(marker, 36);

	VideoCapture cap(0);
	while (true) {
		Mat frame;
		cap >> frame;

		Mat gray;
		cvtColor(frame, gray, COLOR_BGRA2GRAY);
		vector<ArucoResult> ars = detectArucos(gray, dict, 1);
		//Utils::drawArucos(frame, ars);
		Utils::drawAxisWithPose(frame, ars, dict);

		imshow("frame", frame);
		int k = waitKey(50);
		if (k >= 0) {
			break;
		}
	}
}