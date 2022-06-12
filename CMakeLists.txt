# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.13)
# プロジェクト名と使用する言語を設定
project(comarray_so CXX)
#インクルードディレクトリの指定
include_directories(/usr/include /usr/local/include /usr/local/fftw3/include)
# complex_array_classなる共有ライブラリをcomplex_array_class.cppから作成、
add_library(complex_array_class SHARED complex_array_class.cpp)