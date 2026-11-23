cmake_minimum_required(VERSION 3.14)
project(sakhiya-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(GGML_CUDA   "Enable ggml CUDA backend"   OFF)
option(GGML_VULKAN "Enable ggml Vulkan backend" OFF)
option(GGML_METAL  "Enable ggml Metal backend"  OFF)
option(GGML_OPENMP "Enable OpenMP in ggml"      ON)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static link ggml into standalone sakhiya-cli" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml" ggml_build EXCLUDE_FROM_ALL)
elseif(EXISTS "/tmp/ggml/CMakeLists.txt")
    add_subdirectory("/tmp/ggml" ggml_build EXCLUDE_FROM_ALL)
else()
    include(FetchContent)
    FetchContent_Declare(
        ggml
        GIT_REPOSITORY https://github.com/ggerganov/ggml.git
        GIT_TAG        master
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(ggml)
endif()

find_package(OpenMP)

add_executable(sakhiya-cli cpp/sakhiya_cli.cpp)
target_include_directories(sakhiya-cli PRIVATE cpp)
target_link_libraries(sakhiya-cli PRIVATE ggml ggml-base ggml-cpu)

if(OpenMP_CXX_FOUND)
    target_link_libraries(sakhiya-cli PRIVATE OpenMP::OpenMP_CXX)
endif()

if(GGML_CUDA)
    target_compile_definitions(sakhiya-cli PRIVATE GGML_USE_CUDA)
    if(TARGET ggml-cuda)
        target_link_libraries(sakhiya-cli PRIVATE ggml-cuda)
    endif()
endif()

if(GGML_VULKAN)
    target_compile_definitions(sakhiya-cli PRIVATE GGML_USE_VULKAN)
    if(TARGET ggml-vulkan)
        target_link_libraries(sakhiya-cli PRIVATE ggml-vulkan)
    endif()
endif()

if(GGML_METAL)
    target_compile_definitions(sakhiya-cli PRIVATE GGML_USE_METAL)
    if(TARGET ggml-metal)
        target_link_libraries(sakhiya-cli PRIVATE ggml-metal)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sakhiya-cli PRIVATE -O3 -march=native)
endif()
