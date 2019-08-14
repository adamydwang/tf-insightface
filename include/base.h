#ifndef BASE_H
#define BASE_H
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>

void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M);

#endif
