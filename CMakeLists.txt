cmake_minimum_required(VERSION 3.18)
project(adasigboost LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_adasigboost_cpp
    adasigboost/csrc/bindings.cpp
    adasigboost/csrc/hist_tree.cpp
    adasigboost/csrc/coord_descent.cpp
)

target_include_directories(_adasigboost_cpp PRIVATE adasigboost/csrc)
target_compile_options(_adasigboost_cpp PRIVATE -O3 -Wall -Wextra)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_adasigboost_cpp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_adasigboost_cpp PRIVATE ADASIGBOOST_USE_OPENMP=1)
endif()

install(TARGETS _adasigboost_cpp DESTINATION adasigboost)
