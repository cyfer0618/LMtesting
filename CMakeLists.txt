cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(LMtesting)

find_package(Torch REQUIRED)

add_executable(LMtesting LMtesting.cpp)
target_link_libraries(LMtesting "${TORCH_LIBRARIES}")
MESSAGE("TORCH LIBS ${TORCH_LIBRARIES}")
set_property(TARGET LMtesting PROPERTY CXX_STANDARD 14)

