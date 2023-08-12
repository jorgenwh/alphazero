cmake_minimum_required(VERSION 3.16)
project(alphazero_C LANGUAGES CXX CUDA)

set(SOURCES 

)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
