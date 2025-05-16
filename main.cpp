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

int main(int argc, char *argv[]) {
  cout << "Hello, World!" << endl;

  string image_path = samples::findFile("./assets/images/480pln.jpg");
  Mat img = imread(image_path, IMREAD_COLOR);

  pyrDown(img, img);
  pyrDown(img, img);

  Mat gray;
  cvtColor(img, gray, COLOR_BGR2GRAY);

  if (gray.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }
  // 9/10, 7/10, 11/10
  // testAdaptiveThreshold(gray);

  adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
                    9, 10);

  int kernel_size = 5;

  // Dylataja z wykorzystaniem wlasnego elementu strukturalnego
  Mat img_dil_kernel;
  Mat morph_kernel =
      getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));

  // Erozja z wykorzystaniem domyslnego elementu strukturalnego
  Mat img_ero_default;
  erode(gray, img_ero_default, Mat());
  dilate(img_ero_default, img_ero_default, Mat());
  imshow("erode-> dilate domyslnego elementu strukturalnego", img_ero_default);

  // Dylataja z wykorzystaniem wlasnego elementu strukturalnego
  Mat img_ero_kernel;
  erode(gray, img_ero_kernel, morph_kernel);
  dilate(img_ero_kernel, img_ero_kernel, morph_kernel);
  imshow("Erozja->dilate z wykorzystaniem wlasnego elementu strukturalnego",
         img_ero_kernel);

  int k = waitKey(0);
  return 0;
}
