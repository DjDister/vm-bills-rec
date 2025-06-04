#include "main.h"
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct Template {
  int value;
  string label;
  vector<string> fileNames;
};

struct FoundNumber {
  int value;
  Point location;
  float distanceFromLeftTopCorner;
};

bool compareDistances(FoundNumber a, FoundNumber b) {
  return a.distanceFromLeftTopCorner < b.distanceFromLeftTopCorner;
};

struct CalculatedDistance {
  int xDistance;
  int yDistance;
};

CalculatedDistance calculateDistance(Point a, Point b) {
  CalculatedDistance distance;
  distance.xDistance = abs(a.x - b.x);
  distance.yDistance = abs(a.y - b.y);
  return distance;
}

vector<Rect> drawBanknoteBox(Mat &baseImg, Mat &contoursImg) {
  adaptiveThreshold(contoursImg, contoursImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY, 13, 5);
  imshow("Adaptive Threshold", contoursImg);
  Mat erozjaKernelKontury = getStructuringElement(MORPH_RECT, Size(6, 6));
  morphologyEx(contoursImg, contoursImg, MORPH_OPEN, erozjaKernelKontury);
  imshow("Erozja", contoursImg);
  Mat dyltacjaKernelKontury = getStructuringElement(MORPH_RECT, Size(5, 5));
  morphologyEx(contoursImg, contoursImg, MORPH_CLOSE, dyltacjaKernelKontury);
  vector<vector<Point>> contours;
  imshow("Dyltacja", contoursImg);
  findContours(contoursImg, contours, {}, RETR_LIST, CHAIN_APPROX_SIMPLE);

  vector<Rect> boxesWithNumbers;

  int MIN_CONTOUR_SIZE = 100;

  for (int i = 0; i < contours.size(); i++) {
    if (contours[i].size() < MIN_CONTOUR_SIZE) {
      continue;
    }
    Rect boundingBox = boundingRect(contours[i]);

    if (boundingBox.width >= baseImg.cols * 0.99 &&
        boundingBox.height >= baseImg.rows * 0.99) {
      continue;
    }

    rectangle(baseImg, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 0),
              2);

    Point RectTopRight(boundingBox.x + boundingBox.width, boundingBox.y - 5);

    Point smallRectSecondPoint(boundingBox.x + boundingBox.width / 2.5,
                               RectTopRight.y + boundingBox.height * 0.18);

    Rect boxWithNumber = Rect(RectTopRight, smallRectSecondPoint);

    rectangle(baseImg, RectTopRight, smallRectSecondPoint, Scalar(255, 0, 0),
              2);

    boxesWithNumbers.push_back(boxWithNumber);
  }

  return boxesWithNumbers;
}

vector<Template> number_templates =
    vector<Template>{{0,
                      "zero",
                      {
                          "0_template5.png",
                          "0_template6.png",
                          "0_template7.png",
                      }},

                     {1, "one", {"1_template3.png"}},
                     {2,
                      "two",
                      {
                          "2_template3.png",
                          "2_template5.png",
                          "2_template6.png",
                      }},
                     {5,
                      "five",
                      {
                          "5_template2.png",
                      }}};

float templateMatchThreshold = 0.70f;
int distanceBetweenNumbersToMerge = 50;
vector<int> rotation_angles = {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                               1,   2,  3,  4,  5,  6,  7,  8,  9,  10};

int main(int argc, char *argv[]) {
  int totalValue = 0;

  string image_path = samples::findFile("./assets/images/280pln.jpg");
  Mat img = imread(image_path, IMREAD_COLOR);

  resize(img, img, Size(768, 1024));
  imshow("Resize", img);

  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);
  bitwise_not(gray, gray);
  imshow("Gray", gray);

  Mat contoursImg = gray.clone();
  Mat baseImg = img.clone();

  vector<Rect> boxesWithNumbers = drawBanknoteBox(baseImg, contoursImg);
  Mat blackBaseImg = Mat::zeros(baseImg.size(), baseImg.type());
  for (int i = 0; i < boxesWithNumbers.size(); ++i) {
    imshow("Box with number", baseImg(boxesWithNumbers[i]));
    baseImg(boxesWithNumbers[i]).copyTo(blackBaseImg(boxesWithNumbers[i]));
  }
  imshow("BlackImg with boxes", blackBaseImg);

  Mat gray_black = blackBaseImg.clone();
  cvtColor(gray_black, gray_black, COLOR_BGR2GRAY);
  imshow("Gray Black", gray_black);

  adaptiveThreshold(gray_black, gray_black, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY, 41, 5);
  imshow("Adaptive Threshold na img z boxami", gray_black);

  Mat erozja_kernel2 = getStructuringElement(MORPH_RECT, Size(4, 4));
  morphologyEx(gray_black, gray_black, MORPH_OPEN, erozja_kernel2);
  imshow("Erozja na img z boxami", gray_black);
  Mat dylatacj_kernel2 = getStructuringElement(MORPH_RECT, Size(4, 4));
  morphologyEx(gray_black, gray_black, MORPH_CLOSE, dylatacj_kernel2);

  imshow("dylatacja na img z boxami", gray_black);
  bitwise_not(gray_black, gray_black);
  imshow("Bitwise Not", gray_black);
  vector<FoundNumber> found_numbers;

  for (int i = 0; i < number_templates.size(); ++i) {

    vector<Point> unique_points;

    int min_distance_between_points = 10;
    vector<string> rotatedFiles;
    for (int j = 0; j < number_templates[i].fileNames.size(); ++j) {
      cout << "Processing template: " << number_templates[i].fileNames[j]
           << endl;

      for (int k = 0; k < rotation_angles.size(); k++) {
        string nameDuplicate = number_templates[i].fileNames[j].substr(
            0, number_templates[i].fileNames[j].length() - 4);
        rotatedFiles.push_back(nameDuplicate.erase(nameDuplicate.length(), 4) +
                               "_rot_" + to_string(rotation_angles[k]) +
                               ".png");
      }
    }
    number_templates[i].fileNames.insert(number_templates[i].fileNames.end(),
                                         rotatedFiles.begin(),
                                         rotatedFiles.end());

    for (int j = 0; j < number_templates[i].fileNames.size(); ++j) {
      cout << "Processing template: " << number_templates[i].fileNames[j]
           << endl;

      string template_path = samples::findFile(
          "./assets/images/templates/" + to_string(number_templates[i].value) +
          "/" + number_templates[i].fileNames[j]);

      Mat number_template = imread(template_path, IMREAD_GRAYSCALE);
      if (number_template.empty()) {
        cout << "failed to load img: " << template_path << endl;
        return 1;
      }

      Mat result;
      matchTemplate(gray_black, number_template, result, TM_CCOEFF_NORMED);

      Mat locations;
      findNonZero(result >= templateMatchThreshold, locations);
      // cout << "locations.total()" << locations.total() << endl;

      for (int k = 0; k < locations.total(); ++k) {

        Point pt = locations.at<Point>(k);
        unique_points.push_back(pt);
      }
    }
    // cout << "unique_points.size(): " << unique_points.size() << endl;
    for (int k = 0; k < unique_points.size(); ++k) {
      Point pt = unique_points[k];
      bool is_unique = true;
      for (int l = k + 1; l < unique_points.size(); ++l) {
        Point pt2 = unique_points[l];
        if (abs(pt.x - pt2.x) < min_distance_between_points &&
            abs(pt.y - pt2.y) < min_distance_between_points) {
          is_unique = false;
          break;
        }
      }
      if (is_unique) {
        FoundNumber found_number;
        found_number.value = number_templates[i].value;
        found_number.location = pt;
        found_number.distanceFromLeftTopCorner =
            sqrt(pow(pt.x, 2) + pow(pt.y, 2));
        found_numbers.push_back(found_number);
        rectangle(baseImg, pt, Point(pt.x + 30, pt.y + 40), Scalar(0, 0, 255),
                  2);
        putText(baseImg, number_templates[i].label, Point(pt.x + 60, pt.y + 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 0.5);
      }
    }
  }

  sort(found_numbers.begin(), found_numbers.end(), compareDistances);
  cout << "found_numbers.size(): " << found_numbers.size() << endl;

  vector<vector<FoundNumber>> merged_found_numbers;
  merged_found_numbers.push_back(vector<FoundNumber>{found_numbers[0]});
  int i = 1;
  for (i; i < found_numbers.size(); i++) {

    bool isMerged = false;
    for (int j = 0; j < merged_found_numbers.size(); j++) {
      int sizeBefore = merged_found_numbers[j].size();
      int k = 0;
      for (k; k < merged_found_numbers[j].size(); k++) {
        CalculatedDistance distance = calculateDistance(
            found_numbers[i].location, merged_found_numbers[j][k].location);

        if (distance.xDistance < distanceBetweenNumbersToMerge &&
            distance.yDistance < distanceBetweenNumbersToMerge) {
          merged_found_numbers[j].push_back(found_numbers[i]);

          isMerged = true;
          break;
        }
      }
    }
    if (!isMerged) {
      merged_found_numbers.push_back(vector<FoundNumber>{found_numbers[i]});
    }
  }

  vector<string> merged_found_numbers_labels;

  for (int i = 0; i < merged_found_numbers.size(); ++i) {
    string label = "";
    for (int j = 0; j < merged_found_numbers[i].size(); ++j) {
      label += to_string(merged_found_numbers[i][j].value);
    }
    merged_found_numbers_labels.push_back(label);
    totalValue = totalValue + stoi(label);
    cout << "merged found number: " << label << endl;
  }

  imshow("after match tempalte", baseImg);
  cout << "Total value: " << totalValue << endl;
  int k = waitKey(0);
  return 0;
}