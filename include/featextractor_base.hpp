#ifndef __FEATEXTRACTOR_BASE_HPP__
#define __FEATEXTRACTOR_BASE_HPP__
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>
#include <fstream>
#include <base.h>
#include <face_preprocess.hpp>

class FeatExtractorBase {
  public:
    FeatExtractorBase() {}
    FeatExtractorBase(const std::string& modelf) {
      m_modelf = modelf;
    }
    virtual ~FeatExtractorBase(){
      for (unsigned int i = 0; i < m_ivals.size(); ++i) {
        if (m_ivals[i]) {
          TF_DeleteTensor(m_ivals[i]);
        }
      }
      for (unsigned int i = 0; i < m_ovals.size(); ++i) {
        if (m_ovals[i]) {
          TF_DeleteTensor(m_ovals[i]);
        }
      }
      if (m_status == NULL) {
        m_status = TF_NewStatus();
      }
      if (m_sess) {
        TF_CloseSession(m_sess, m_status);
        TF_DeleteSession(m_sess, m_status);
      }
      if (m_graph) {
        TF_DeleteGraph(m_graph);
      }
      TF_DeleteStatus(m_status);
    }
    virtual int setup()=0;
    virtual int extract(cv::Mat& image, cv::Rect& rect, std::vector<float>& landmark, std::vector<float>& feat)=0;

  protected:
    std::string m_modelf;
    std::string m_device;
    int m_width;
    int m_height;
    int m_margin;
    std::vector<std::string> m_inputs;
    std::vector<std::string> m_outputs;

    TF_ImportGraphDefOptions* m_iopts;
    TF_SessionOptions* m_sopts;
    TF_Graph*   m_graph;
    TF_Session* m_sess;
    TF_Status*  m_status;
    std::vector<TF_Output> m_iops;
    std::vector<TF_Output> m_oops;
    std::vector<TF_Tensor*> m_ivals;
    std::vector<TF_Tensor*> m_ovals;

    virtual void norm(std::vector<float>& feat) {
      if (feat.empty()) {
        return;
      }
      double total;
      for (size_t i = 0; i < feat.size(); ++i) {
        total += feat[i] * feat[i];
      }
      total = sqrt(total);
      for (size_t i = 0; i < feat.size(); ++i) {
        feat[i] /= total;
      }
    }


  void imcrop(cv::Mat& srcImage, cv::Rect& face, int margin, cv::Mat& roiImage) {
    int half = margin / 2;
    cv::Rect rect;
    rect.x = face.x - half;
    rect.y = face.y - half;
    rect.width = face.width + margin;
    rect.height = face.height + margin;
    //cv::Mat srcImage = image.clone();
    int crop_x1 = cv::max(0, rect.x);
    int crop_y1 = cv::max(0, rect.y);
    int crop_x2 = cv::min(srcImage.cols, rect.x + rect.width);
    int crop_y2 = cv::min(srcImage.rows, rect.y + rect.height);

    int left_x = (-rect.x);
    int top_y = (-rect.y);
    int right_x = rect.x + rect.width - srcImage.cols;
    int down_y = rect.y + rect.height - srcImage.rows;
    roiImage = srcImage(cv::Rect(crop_x1, crop_y1, (crop_x2 - crop_x1), (crop_y2 - crop_y1)));

    if (top_y > 0 || down_y > 0 || left_x > 0 || right_x > 0) {
      left_x = (left_x > 0 ? left_x : 0);
      right_x = (right_x > 0 ? right_x : 0);
      top_y = (top_y > 0 ? top_y : 0);
      down_y = (down_y > 0 ? down_y : 0);
      cv::copyMakeBorder(roiImage, roiImage, top_y, down_y, left_x, right_x, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
  }

    virtual int loadFile(const std::string& modelf, std::vector<char>& buffer) {
      std::ifstream fi(modelf, std::ios::binary | std::ios::in);
      if (!fi.is_open()) {
        std::cerr << modelf << ": file open failed" << std::endl;
        return -1;
      }
      fi.seekg(0, std::ios::end);
      int size = fi.tellg();
      fi.seekg(0, std::ios::beg);
      buffer.resize(size);
      fi.read(buffer.data(), size);
      fi.close();
      return 0;
    }
    virtual TF_Session* loadGraph(const std::string& modelf) {
      std::vector<char> buffer;
      if (this->loadFile(modelf, buffer) < 0) {
        std::cerr << "load file fail" << std::endl;
        return nullptr;
      }
      std::cout << "load file done, size=" << buffer.size() << std::endl;
      TF_Buffer graphdef = {buffer.data(), buffer.size(), nullptr};
      TF_ImportGraphDefOptionsSetPrefix(m_iopts, "");
      TF_GraphImportGraphDef(m_graph, &graphdef, m_iopts, m_status);
      if (TF_GetCode(m_status) != TF_OK) {
        std::cerr << "graph import def failed !," << TF_Message(m_status) << std::endl;
        return nullptr;
      }
      TF_Session* sess = TF_NewSession(m_graph, m_sopts, m_status);
      if (TF_GetCode(m_status) != TF_OK) {
        std::cerr << "new session fail!," << TF_Message(m_status) << std::endl;
        return nullptr;
      }
      return sess;
    }


  bool isPoseProper(std::vector<float>& landmark) {
    double eyeDis = (landmark[2] - landmark[0]) * (landmark[2] - landmark[0]) + (landmark[3] - landmark[1]) * (landmark[3] - landmark[1]);
    double up = (landmark[3] + landmark[1]) * 0.5;
    double down = (landmark[7] + landmark[9]) * 0.5;
    double vertical = (down - up) * (down - up);
    if (vertical / eyeDis > 5 || eyeDis / vertical > 5) {
      std::cout << "Pose not proper: " << eyeDis  << ", " << vertical << std::endl;
      return false;
    }
    return true;
  }

  void align(cv::Mat& img, std::vector<float>& landmark, cv::Mat& ret) {
    float dst[10] = {38.2946, 73.5318, 56.0252, 41.5493, 70.7299,
                     51.6963, 51.5014, 71.7366, 92.3655, 92.2041};
    float src[10];
    for (int i = 0; i < 5; ++i) {
      src[i] = landmark[2*i];
      src[i+5] = landmark[2*i+1];
    }
    float M[6];
    getAffineMatrix(src, dst, M);
    cv::Mat m(2, 3, CV_32F);
    for (int i = 0; i < 6; ++i) {
      m.at<float>(i) = M[i];
    }
    cv::warpAffine(img, ret, m, cv::Size(m_width, m_height));
    
  }
};

#endif
