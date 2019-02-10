#include <insightface.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
InsightFace::~InsightFace() {
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
static void dummy_deallocator(void* data, size_t len, void* arg) {

}

int InsightFace::setup() {
  m_iopts = TF_NewImportGraphDefOptions();
  m_sopts = TF_NewSessionOptions();
  m_status = TF_NewStatus();
  m_graph = TF_NewGraph();
  m_sess = this->loadGraph(m_modelf);
  if (m_sess == NULL) {
    std::cerr << "load graph fail" << std::endl;
    return -1;
  }
  m_iops.resize(m_inputs.size());
  m_oops.resize(m_outputs.size());
  m_ivals.resize(m_inputs.size());
  m_ovals.resize(m_outputs.size());
  for (unsigned int i = 0; i < m_inputs.size(); ++i) {
    m_iops[i] = {TF_GraphOperationByName(m_graph, m_inputs[i].c_str()), 0};
    if (m_iops[i].oper == nullptr) {
      std::cerr << "input operation not found: " << m_inputs[i] << std::endl;
      return -1;
    }
  }
  for (unsigned int i = 0; i < m_outputs.size(); ++i) {
    m_oops[i] = {TF_GraphOperationByName(m_graph, m_outputs[i].c_str()), 0};
    if (m_oops[i].oper == nullptr) {
      std::cerr << "output operation not found: " << m_outputs[i] << std::endl;
      return -1;
    }
  }
  int64_t dim[1] = {1};
  m_ivals[1] = TF_NewTensor(TF_FLOAT, dim, 1, &m_dropout, sizeof(float), dummy_deallocator, nullptr); 
  return 0;
}

int InsightFace::extract(cv::Mat& face, std::vector<float>& feat) {
  if (face.empty() || face.channels() != 3) {
    std::cerr << "input image not valid" << std::endl;
    return -1;
  }
  cv::Mat resized;
  face.convertTo(resized, CV_32FC3);
  cv::resize(resized, resized, cv::Size(m_width, m_height));
  const int64_t dim[4] = {1, m_width, m_height, 3};
  m_ivals[0] = TF_NewTensor(TF_FLOAT, dim, 4, resized.ptr<float>(), sizeof(float) * m_width * m_height * 3, dummy_deallocator, nullptr);
  TF_SessionRun(m_sess, nullptr, m_iops.data(), m_ivals.data(), m_iops.size(), m_oops.data(), m_ovals.data(), m_oops.size(), nullptr, 0, nullptr, m_status);
  if (TF_GetCode(m_status) != TF_OK) {
    std::cerr << "session run fail" << std::endl;
    return -1;
  }
  feat.resize(TF_Dim(m_ovals[0], 1));
  const float* raw = (const float*)TF_TensorData(m_ovals[0]);
  for (unsigned int i = 0; i < feat.size(); ++i) {
    feat[i] = raw[i];
  }
  for (unsigned int i = 0; i < m_ovals.size(); ++i) {
    TF_DeleteTensor(m_ovals[i]);
    m_ovals[i] = nullptr;
  }
  TF_DeleteTensor(m_ivals[0]);
  m_ivals[0] = nullptr;
  return 0;
}


int InsightFace::loadFile(const std::string& modelf, std::vector<char>& buffer) {
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

TF_Session* InsightFace::loadGraph(const std::string& modelf) {
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
