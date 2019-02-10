#ifndef __INSIGHTFACE_BASE_HPP__
#define __INSIGHTFACE_BASE_HPP__
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class InsightFaceBase {
  public:
    InsightFaceBase() {}
    InsightFaceBase(const std::string& modelf, 
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs, 
      int width, int height, float dropout,
      const std::string& device) {
      m_modelf = modelf;
      m_inputs = inputs;
      m_outputs = outputs;
      m_device = device;
      m_width = width;
      m_height = height;
      m_dropout = dropout;
    } 
    virtual ~InsightFaceBase(){}
    virtual int setup()=0;
    virtual int extract(cv::Mat& face, std::vector<float>& feat)=0;

  protected:
    std::string m_modelf;
    std::string m_device;
    int m_width;
    int m_height;
    float m_dropout;
    std::vector<std::string> m_inputs;
    std::vector<std::string> m_outputs;
};

#endif
