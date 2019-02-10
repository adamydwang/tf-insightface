#ifndef __INSIGHTFACE_HPP__
#define __INSIGHTFACE_HPP__
#include <insightface_base.hpp>
#include <tensorflow/c/c_api.h>
class InsightFace : public InsightFaceBase {
  public:
    InsightFace(const std::string& modelf) : InsightFaceBase() {
      m_inputs.push_back("img_inputs");
      m_inputs.push_back("dropout_rate");
      m_outputs.push_back("resnet_v1_50/E_BN2/Identity");
      m_dropout = 0.5;
      m_width = 112;
      m_height = 112;
      m_device = "";
      m_modelf = modelf;
    }
    ~InsightFace();
    int setup();
    int extract(cv::Mat& face, std::vector<float>& feat);

  private:
    TF_ImportGraphDefOptions* m_iopts;
    TF_SessionOptions* m_sopts;
    TF_Graph*   m_graph;
    TF_Session* m_sess;
    TF_Status*  m_status;
    std::vector<TF_Output> m_iops;
    std::vector<TF_Output> m_oops;
    std::vector<TF_Tensor*> m_ivals;
    std::vector<TF_Tensor*> m_ovals;

  private:
    int loadFile(const std::string& modelf, std::vector<char>& buffer);
    TF_Session* loadGraph(const std::string& modelf);
};

#endif
