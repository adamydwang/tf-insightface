#ifndef __FACE_PREPROCESS_HPP__
#define __FACE_PREPROCESS_HPP__

#include<opencv2/opencv.hpp>


namespace FacePreprocess {

    cv::Mat meanAxis0(const cv::Mat &src);

    cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B);

    cv::Mat varAxis0(const cv::Mat &src);

    int MatrixRank(cv::Mat M);

    cv::Mat similarTransform(cv::Mat src,cv::Mat dst);

}
#endif
