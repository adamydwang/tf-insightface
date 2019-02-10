#include <insightface.hpp>
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
  std::string modelf = "./models/insightface.pb";
  InsightFace iface(modelf);
  if (iface.setup() != 0) {
    std::cout << "setup fail" << std::endl;
    return -1;
  }
  std::cout << "setup done" << std::endl;
  cv::Mat face = cv::imread(image);
  std::cout << "read image done" << std::endl;
  std::vector<float> feat;
  if (iface.extract(face, feat) != 0) {
    std::cout << "extract fail" << std::endl;
    return -1;
  }
  unsigned long start = getCurTime();
  for (int i = 0; i < 10; ++i) {
    if (iface.extract(face, feat) != 0) {
      std::cout << "extract feat fail" << std::endl;
      return -1;
    }
  }
  unsigned long end = getCurTime();
  std::cout << "Average Time: " << (end - start) / 10 / 1000 << " ms" << std::endl;
  for (int i = 0; i < feat.size(); ++i) {
    std::cout << feat[i] << ",";
  }
}
