unzip 2.4.13.5.zip
curpath=`pwd`
cd opencv-2.4.13.5
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=${curpath}/opencv ..
make -j4
make install
cd ../../
rm -rf opencv-2.4.13.5
