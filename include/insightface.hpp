#ifndef __INSIGHTFACE_HPP__
#define __INSIGHTFACE_HPP__
#include <featextractor_base.hpp>
#include <tensorflow/c/c_api.h>
class InsightFace : public FeatExtractorBase {
  public:
    InsightFace(const std::string& modelf) : FeatExtractorBase(modelf) {
      //m_inputs.push_back("img_inputs");
      //m_inputs.push_back("dropout_rate");
      //m_outputs.push_back("resnet_v1_50/E_BN2/Identity");
      //m_inputs.push_back("input_place");
      //m_outputs.push_back("resnet_v1_50/Mean");
      m_inputs.push_back("data");
      m_outputs.push_back("fc1/add_1");
      m_width = 112;
      m_height = 112;
      m_device = "";
      m_margin = 22;
      m_dropout = 1.0;
    }
    ~InsightFace() {}
    int setup();
    int extract(cv::Mat& image, cv::Rect& rect, std::vector<float>& landmark, std::vector<float>& feat);

  private:
    void preprocess(cv::Mat& image, std::vector<float>& landmark, cv::Mat& face);
    float m_dropout;
};

#endif
