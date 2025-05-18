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

vector<Template> number_templates = vector<Template>{
    {0, "zero", {"0_template1.png", "0_template2.png"}},

    {2, "two", {"2_template1.png"}},

};

float templateMatchThreshold = 0.6;

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

  for (int i = 0; i < number_templates.size(); ++i) {
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

      imshow("number template", number_template);

      Mat result;
      matchTemplate(processed_img, number_template, result, TM_CCOEFF_NORMED);

      Mat locations;
      findNonZero(result >= templateMatchThreshold, locations);

      for (int k = 0; k < locations.total(); ++k) {
        Point pt = locations.at<Point>(k);
        rectangle(
            baseImg, pt,
            Point(pt.x + number_template.cols, pt.y + number_template.rows),
            Scalar(0, 0, 255), 2);
      }
    }
  }
  imshow("after match tempalte", baseImg);

  int k = waitKey(0);
  return 0;
}