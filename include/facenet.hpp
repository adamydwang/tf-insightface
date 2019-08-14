#ifndef __FACENET_HPP__
#define __FACENET_HPP__
#include <featextractor_base.hpp>
#include <tensorflow/c/c_api.h>
class FaceNet : public FeatExtractorBase {
  public:
    FaceNet(const std::string& modelf) : FeatExtractorBase(modelf) {
      m_inputs.push_back("input");
      m_inputs.push_back("phase_train");
      m_outputs.push_back("embeddings");
      m_width = 160;
      m_height = 160;
      m_device = "";
      m_margin = 44;
    }
    ~FaceNet() {}
    int setup();
    int extract(cv::Mat& image, cv::Rect& rect, std::vector<float>& landmark, std::vector<float>& feat);

  private:
    void preprocess(cv::Mat& image, cv::Rect& rect, std::vector<float>& landmark, cv::Mat& face);
};

#endif
