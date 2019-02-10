cmake_minimum_required(VERSION 3.5)
project(insightface)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_DEBUG_FLAGS "${CMAKE_CXX_DEBUG_FLAGS} -g -std=c++11 -W -fPIC")
set(CMAKE_BUILD_TYPE "debug")
add_definitions(-W)
add_definitions(-fPIC)
add_definitions(-std=c++11)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(THIRDPARTY ${PROJECT_SOURCE_DIR}/thirdparty)
set(TF_PATH ${THIRDPARTY}/tensorflow)
set(TF_INCLUDE_DIRS ${TF_PATH}/include)
set(TF_LIBRARY_DIRS ${TF_PATH}/lib)
set(TF_LDFLAGS tensorflow tensorflow_framework)

set(ENV{PKG_CONFIG_PATH} "${PROJECT_SOURCE_DIR}/thirdparty/opencv/lib/pkgconfig")
include(FindPkgConfig)
pkg_check_modules(CV REQUIRED opencv)


aux_source_directory(${PROJECT_SOURCE_DIR}/src SRCS)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${CV_INCLUDE_DIRS}
                    ${TF_INCLUDE_DIRS})

link_directories(${CV_LIBRARY_DIRS}
                 ${TF_LIBRARY_DIRS})


add_library(${PROJECT_NAME} SHARED ${SRCS})
add_library(${PROJECT_NAME}_static STATIC ${SRCS})
add_executable(demo ${SRCS} ${PROJECT_SOURCE_DIR}/src/demo.cpp)


target_link_libraries(${PROJECT_NAME} 
                      ${TF_LDFLAGS}
                      ${CV_LDFLAGS})

target_link_libraries(${PROJECT_NAME}_static
                    ${TF_LDFLAGS}
                    ${CV_LDFLAGS})

target_link_libraries(demo ${TF_LDFLAGS} ${CV_LDFLAGS})

set_target_properties(${PROJECT_NAME}_static 
                      PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

set(OUTPUT ${PROJECT_SOURCE_DIR}/output)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/*)
install(FILES ${HEADERS} DESTINATION ${OUTPUT}/include)
file(GLOB TFLIBS  ${TF_LIBRARY_DIRS}/*)
install(FILES ${TFLIBS} DESTINATION ${OUTPUT}/lib)
install(TARGETS ${PROJECT_NAME} DESTINATION ${OUTPUT}/lib)
install(TARGETS ${PROJECT_NAME}_static DESTINATION ${OUTPUT}/lib)

