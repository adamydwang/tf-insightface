#include <featextractor_factory.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

unsigned long getCurTime() {
  struct timeval tv;
  unsigned long ts;
  gettimeofday(&tv, NULL);
  ts = tv.tv_sec * 1000000 + tv.tv_usec;
  return ts;
}

int main() {
  std::string image = "./images/test.jpg";
  //std::string modelf = "./models/20180402-114759.pb";
  //std::string modelf = "./models/insightface.pb";
  std::string modelf = "./models/tfmodel.pb";
  FeatExtractorBase* iface = FeatExtractorFactory::create(InsightFaceMode, modelf);
  if (iface->setup() != 0) {
    std::cout << "setup fail" << std::endl;
    return -1;
  }
  std::cout << "setup done" << std::endl;
  cv::Mat face = cv::imread(image);
  std::cout << "read image done" << std::endl;
  std::vector<float> feat;
  cv::Rect rect(0, 0, face.cols, face.rows);
  std::vector<float> landmarks(10);
  if (iface->extract(face, rect, landmarks, feat) != 0) {
    std::cout << "extract fail" << std::endl;
    return -1;
  }
  unsigned long start = getCurTime();
  for (int i = 0; i < 10; ++i) {
    if (iface->extract(face, rect, landmarks, feat) != 0) {
      std::cout << "extract feat fail" << std::endl;
      return -1;
    }
  }
  unsigned long end = getCurTime();
  std::cout << "Average Time: " << (end - start) / 10 / 1000 << " ms" << std::endl;
  for (size_t i = 0; i < feat.size(); ++i) {
    if (i % 128 == 0) {
      std::cout << std::endl;
    }
    std::cout << feat[i] << ",";
  }
  std::cout << std::endl;
  std::cout << "output dimesion: " << feat.size() << std::endl;
}
