#include "main.h"
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void testAdaptiveThreshold(Mat gray) {
  Mat processed_gray;
  for (int block_size = 3; block_size <= 55; block_size += 2) {
    for (int c_value = -30; c_value <= 30; c_value += 5) {
      gray.copyTo(processed_gray);
      adaptiveThreshold(processed_gray, processed_gray, 255,
                        ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block_size,
                        c_value);

      string window_name =
          "BlockSize=" + to_string(block_size) + " C=" + to_string(c_value);
      imshow(window_name, processed_gray);
      cout << window_name << endl;
      int k = waitKey(0);
      if (k == 27) {
        destroyAllWindows();

        return;
      }
      destroyWindow(window_name);
    }
  }
}

void testMorphology(Mat base_img) {
  Mat current_img;
  for (int open_k_size = 1; open_k_size <= 7; ++open_k_size) {
    for (int close_k_size = 1; close_k_size <= 7; ++close_k_size) {
      if (open_k_size == 0 || close_k_size == 0)
        continue;

      base_img.copyTo(current_img);

      Mat open_kernel =
          getStructuringElement(MORPH_RECT, Size(open_k_size, open_k_size));
      morphologyEx(current_img, current_img, MORPH_OPEN, open_kernel);

      Mat close_kernel =
          getStructuringElement(MORPH_RECT, Size(close_k_size, close_k_size));
      morphologyEx(current_img, current_img, MORPH_CLOSE, close_kernel);

      string window_name = "OpenKernel: " + to_string(open_k_size) + "x" +
                           to_string(open_k_size) +
                           " CloseKernel: " + to_string(close_k_size) + "x" +
                           to_string(close_k_size);
      imshow(window_name, current_img);

      int k = waitKey(0);
      if (k == 27) { // ESC
        destroyAllWindows();

        return;
      }
      destroyWindow(window_name);
    }
  }
}

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

struct FoundBanknote {
  int value;
  string label;
  vector<FoundNumber> digits;
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

vector<Template> number_templates = vector<Template>{
    {0, "zero", {"0_template1.png", "0_template2.png", "0_template3.png"}},
    {1, "one", {"1_template1.png"}},
    {2, "two", {"2_template1.png", "2_template2.png"}},

    {5, "one", {"5_template1.png"}},
};

float templateMatchThreshold = 0.74;
int distanceBetweenNumbersToMerge = 50;

int main(int argc, char *argv[]) {
  cout << "Hello, World!" << endl;

  string image_path = samples::findFile("./assets/images/480pln.jpg");
  Mat img = imread(image_path, IMREAD_COLOR);

  pyrDown(img, img);
  pyrDown(img, img);

  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);
  bitwise_not(gray, gray);

  if (gray.empty()) {
    cout << "failed to load img: " << image_path << endl;
    return 1;
  }
  // 9/10, 7/10, 11/10
  // testAdaptiveThreshold(gray);

  adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
                    29, 5);

  Mat processed_img;
  gray.copyTo(processed_img);

  // 4/4 2/2
  // testMorphology(processed_img);

  Mat erozja_kernel = getStructuringElement(MORPH_RECT, Size(4, 4));
  morphologyEx(processed_img, processed_img, MORPH_OPEN, erozja_kernel);

  Mat dylatacj_kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
  morphologyEx(processed_img, processed_img, MORPH_CLOSE, dylatacj_kernel);

  imshow("Erozja->dilate z wykorzystaniem wlasnego elementu strukturalnego",
         processed_img);

  Mat baseImg = img.clone();

  vector<FoundNumber> found_numbers;

  for (int i = 0; i < number_templates.size(); ++i) {

    vector<Point> unique_points;

    int min_distance_between_points = 10;

    for (int j = 0; j < number_templates[i].fileNames.size(); ++j) {
      string template_path = samples::findFile(
          "./assets/images/templates/" + to_string(number_templates[i].value) +
          "/" + number_templates[i].fileNames[j]);

      Mat number_template = imread(template_path, IMREAD_GRAYSCALE);
      if (number_template.empty()) {
        cout << "failed to load img: " << template_path << endl;
        return 1;
      }

      // bitwise_not(number_template, number_template);

      Mat result;
      matchTemplate(processed_img, number_template, result, TM_CCOEFF_NORMED);

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
    // cout << "found number: " << found_numbers[i].value
    //      << " at: " << found_numbers[i].location.x << " "
    //      << found_numbers[i].location.y << endl
    //      << "distance: " << found_numbers[i].distanceFromLeftTopCorner <<
    //      endl;
    // cout << "i: " << i << endl;
    // cout << "found number: " << found_numbers[i].value << endl;
    bool isMerged = false;
    for (int j = 0; j < merged_found_numbers.size(); j++) {
      int sizeBefore = merged_found_numbers[j].size();
      int k = 0;
      for (k; k < merged_found_numbers[j].size(); k++) {
        // cout << "merged_found_numbers[j][k].value"
        //      << merged_found_numbers[j][k].value << endl;
        CalculatedDistance distance = calculateDistance(
            found_numbers[i].location, merged_found_numbers[j][k].location);
        if (distance.xDistance < distanceBetweenNumbersToMerge &&
            distance.yDistance < distanceBetweenNumbersToMerge) {
          merged_found_numbers[j].push_back(found_numbers[i]);
          // cout << "found number: " << found_numbers[i].value
          //      << " merged with: " << merged_found_numbers[j][k].value <<
          //      endl;
          isMerged = true;
          break;
        }
      }
    }
    if (!isMerged) {
      merged_found_numbers.push_back(vector<FoundNumber>{found_numbers[i]});
      // cout << "found number: " << found_numbers[i].value
      //      << " added to new group" << endl;
    }
  }

  vector<string> merged_found_numbers_labels;

  for (int i = 0; i < merged_found_numbers.size(); ++i) {
    string label = "";
    for (int j = 0; j < merged_found_numbers[i].size(); ++j) {
      label += to_string(merged_found_numbers[i][j].value);
    }
    merged_found_numbers_labels.push_back(label);
    cout << "merged found number: " << label << endl;
  }

  imshow("after match tempalte", baseImg);

  int k = waitKey(0);
  return 0;
}