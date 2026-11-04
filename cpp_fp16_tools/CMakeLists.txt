cmake_minimum_required(VERSION 3.16)
project(heteronym_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/utf-8 /W3)
endif()

# 1) fp32 -> fp16 converter and verifier: standard library only, no other dependency.
add_executable(fp16_convert src/fp16_convert.cpp)
add_executable(fp16_verify  src/fp16_verify.cpp)
add_executable(fp16_selftest src/fp16_selftest.cpp)

# 2) benchmark: needs the unpacked ONNX Runtime release (the folder with include/ and lib/).
#    cmake -S . -B build -DONNXRUNTIME_DIR=C:/ort/onnxruntime-win-x64-1.20.1
set(ONNXRUNTIME_DIR "" CACHE PATH "unpacked ONNX Runtime C/C++ release (optional, only for bench_ort)")
if(ONNXRUNTIME_DIR)
  add_executable(bench_ort src/bench_ort.cpp)
  target_include_directories(bench_ort PRIVATE ${ONNXRUNTIME_DIR}/include)
  target_link_directories(bench_ort PRIVATE ${ONNXRUNTIME_DIR}/lib)
  target_link_libraries(bench_ort PRIVATE onnxruntime)
  if(MSVC)
    add_custom_command(TARGET bench_ort POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_DIR}/lib/onnxruntime.dll $<TARGET_FILE_DIR:bench_ort>)
  else()
    set_target_properties(bench_ort PROPERTIES BUILD_RPATH ${ONNXRUNTIME_DIR}/lib INSTALL_RPATH ${ONNXRUNTIME_DIR}/lib)
  endif()
else()
  message(STATUS "ONNXRUNTIME_DIR not set: building only fp16_convert and fp16_verify")
endif()
