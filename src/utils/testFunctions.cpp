#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

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

vector<string>
generateRotatedVersionsForFile(const string &originalFullFilePath,
                               const vector<int> &anglesToRotate) {
  vector<string> newGeneratedFileNames;

  string directoryPath;
  string fileNameWithExt;
  int lastSeparator = originalFullFilePath.find_last_of("/\\");

  if (lastSeparator != string::npos) {
    directoryPath = originalFullFilePath.substr(0, lastSeparator + 1);
    fileNameWithExt = originalFullFilePath.substr(lastSeparator + 1);
  } else {

    directoryPath = "./";
    fileNameWithExt = originalFullFilePath;
  }

  Mat originalImage = imread(originalFullFilePath, IMREAD_GRAYSCALE);
  if (originalImage.empty()) {
    cout << "blad ladowania obrazu " << originalFullFilePath << endl;
    return newGeneratedFileNames;
  }

  for (int angle_idx = 0; angle_idx < anglesToRotate.size(); ++angle_idx) {
    int angle = anglesToRotate[angle_idx];
    if (angle == 0) {
      continue;
    }

    Mat rotatedImage;
    Point2f center((originalImage.cols) / 2.0f, (originalImage.rows) / 2.0f);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);

    Rect2f boundingBox =
        RotatedRect(Point2f(), originalImage.size(), angle).boundingRect2f();
    rotationMatrix.at<double>(0, 2) += boundingBox.width / 2.0 - center.x;
    rotationMatrix.at<double>(1, 2) += boundingBox.height / 2.0 - center.y;

    warpAffine(originalImage, rotatedImage, rotationMatrix, boundingBox.size(),
               INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

    int dotPosition = fileNameWithExt.find_last_of(".");
    string nameWithoutExtension = fileNameWithExt;
    string fileExtension = "";

    if (dotPosition != string::npos) {
      nameWithoutExtension = fileNameWithExt.substr(0, dotPosition);
      fileExtension = fileNameWithExt.substr(dotPosition);
    }

    string newFileName =
        nameWithoutExtension + "_rot_" + to_string(angle) + fileExtension;
    string fullSavePath = directoryPath + newFileName;

    if (imwrite(fullSavePath, rotatedImage)) {
      newGeneratedFileNames.push_back(newFileName);
    } else {
      cout << "blad zapisu " << fullSavePath << endl;
    }
  }
  return newGeneratedFileNames;
}