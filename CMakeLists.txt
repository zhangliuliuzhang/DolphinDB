cmake_minimum_required(VERSION 3.16)
project(label_propagation CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(label_propagation src/main.cpp)
target_link_libraries(label_propagation PRIVATE OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(label_propagation PRIVATE -O3 -march=native -funroll-loops)
endif()
