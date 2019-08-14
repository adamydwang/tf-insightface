#ifndef __FEATEXTRACTOR_FACTORY_HPP__
#define __FEATEXTRACTOR_FACTORY_HPP__
#include <facenet.hpp>
#include <insightface.hpp>
#include <featextractor_base.hpp>

enum FeatExtractorMode {
  FaceNetMode = 0,
  InsightFaceMode = 1
};

class FeatExtractorFactory {
public:
  FeatExtractorFactory() {}
  ~FeatExtractorFactory() {}
  static  FeatExtractorBase* create(int mode, const std::string& modelf) {
    switch(mode) {
      case FaceNetMode:
        return new FaceNet(modelf);
      case InsightFaceMode:
        return new InsightFace(modelf);
    }
    return NULL;
  }
};


#endif
