cmake_minimum_required(VERSION 3.26)
project(specdraft_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Baseline for every translation unit: AVX2 + FMA + F16C, with floating-point
# contraction OFF. Contraction (a*b+c fused into one rounding step) would make the
# engine's arithmetic depend on the compiler, which breaks both the byte-exact
# quantization tests against Python and bit-exact greedy speculative decoding.
function(specdraft_baseline_flags target)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${target} PRIVATE /O2 /arch:AVX2 /fp:precise)
  else()
    target_compile_options(${target} PRIVATE -O2 -mavx2 -mfma -mf16c -ffp-contract=off)
  endif()
endfunction()

add_library(engine_core STATIC
  engine/src/cpu.cpp
  engine/src/quant.cpp
  engine/src/model_file.cpp
  engine/src/model.cpp
  engine/src/threadpool.cpp
)
target_include_directories(engine_core PUBLIC engine/include)
specdraft_baseline_flags(engine_core)

pybind11_add_module(_engine engine/bindings.cpp)
target_link_libraries(_engine PRIVATE engine_core)
specdraft_baseline_flags(_engine)

install(TARGETS _engine DESTINATION specdraft)
