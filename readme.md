## Introduction

- This project refered [tf-insightface](https://github.com/AIInAi/tf-insightface)
- Tensorflow model can be downloaded [here](https://drive.google.com/open?id=1Iw2Ckz_BnHZUi78USlaFreZXylJj7hnP)

## Feature

- v1
  - in support of insightface model
- v2
  - in support of insightface & facenet model
  - add face alignment with higher accuracy, mtcnn output can be linked to tf-insightface directory. 
## How to build it

- `cd thirdparty`
- `wget https://github.com/opencv/opencv/archive/2.4.13.5.zip`
- `bash opencv.sh` 
- `cd ..`
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`

## How to run it

- `put tf model in models directory`
- `./bin/demo`

## Performance

Compared with python verison tf-insightface consuming 118ms, c++ version tf-insightface consumes 87ms, 30ms less on my 8 core cpu machine. 
