cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-cpu-compute LANGUAGES CXX )

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_BUILD_TYPE Debug)
# 设置 SIMD 编译选项

# 包含头文件目录
include_directories(src)

if(NOT TARGET deepx_core)
    add_subdirectory(deepx-core deepx-core)
endif()

# cpu 线性代数库
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake")
find_package(OpenBLAS REQUIRED)
if(OpenBLAS_INCLUDE_DIRS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_directories(${OpenBLAS_LIB_DIR})
elseif(APPLE)
    include_directories(/opt/homebrew/opt/openblas/include)
    link_directories(/opt/homebrew/opt/openblas/lib)
else()
    include_directories(/usr/include/openblas)
endif()

# include(FetchContent)

# # Fetch OpenBLAS from GitHub
# FetchContent_Declare(
#   OpenBLAS
#   GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS
#   GIT_TAG        v0.3.29  # 使用最新的稳定版本或您需要的特定版本
# )

# # 下载并构建 OpenBLAS
# FetchContent_MakeAvailable(OpenBLAS)

# 线程并行
find_package(OpenMP REQUIRED)
# simd
find_package(hwy REQUIRED)
if(HWY_INCLUDE_DIRS)
    include_directories(${HWY_INCLUDE_DIRS})
    link_directories(${HWY_LIB_DIR})
elseif(APPLE)
    include_directories(/opt/homebrew/opt/highway/include)
    link_directories(/opt/homebrew/opt/highway/lib)
endif()
# SIMD: x86→AVX2+SSE4.2, ARM→NEON (Apple Clang auto-detects)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse4.2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.5-a")
endif()
# tensordata 全部活在 kvspace shm 里（tensor_bridge 零拷贝借用），不经 deepx-core 内存池，
# 故不依赖 jemalloc；序列化走 kvspace，不依赖 yaml-cpp/nlohmann-json。
# 第三方依赖仅保留：OpenBLAS（cblas）、Highway（SIMD）、OpenMP（并行）。

# 纯模板头文件库（计算 kernel 实现）
add_library(deepx_cpu INTERFACE)
target_include_directories(deepx_cpu INTERFACE src)
target_link_libraries( deepx_cpu
    INTERFACE
    deepx_core
    ${OpenBLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    hwy::hwy
)
# ── main-runtime：嵌入 kvlang/runtime（C 核心），CPU tensor myrwircaps 就地兑现 ──
# 依赖已安装的 C ABI（/usr/lib + /usr/include）：runtime + rwirext（libkvlang_runtime）、
# kvspace（libkvspace）、layout（libkvlanglayout，仅取 kvlangLayoutFile 符号）。
find_library(KVLANG_RUNTIME_LIB kvlang_runtime PATHS /usr/lib REQUIRED)
find_library(KVSPACE_LIB kvspace PATHS /usr/lib REQUIRED)
find_library(KVLANGLAYOUT_LIB kvlanglayout PATHS /usr/lib REQUIRED)

file(GLOB MAIN_RUNTIME_SOURCES "src/main-runtime/*.cpp")
add_executable(${PROJECT_NAME} ${MAIN_RUNTIME_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE /usr/include)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    deepx_cpu
    ${KVLANG_RUNTIME_LIB}
    ${KVSPACE_LIB}
    ${KVLANGLAYOUT_LIB}
)
