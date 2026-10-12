cmake_minimum_required(VERSION 3.22)
project(upsweep_p2p LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# cstone is header-only for the CPU path
add_library(cstone INTERFACE)
target_include_directories(cstone INTERFACE ${PROJECT_SOURCE_DIR}/extern/sphexa/domain/include)

add_library(upsweep_p2p INTERFACE)
target_include_directories(upsweep_p2p INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(upsweep_p2p INTERFACE cstone MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(test_global_exchange test/test_global_exchange.cpp)
target_link_libraries(test_global_exchange PRIVATE upsweep_p2p)
target_compile_options(test_global_exchange PRIVATE -Wall -Wextra)
