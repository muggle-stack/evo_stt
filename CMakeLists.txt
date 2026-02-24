cmake_minimum_required(VERSION 3.16)
project(STT VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Position Independent Code for all targets (required for RISC-V shared libs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------
# Build Options
# -------------------------------------------------------
# BUILD_STREAM_DEMO: 编译麦克风流式识别示例和 Python 绑定
#
# 需要额外依赖:
#   1. PortAudio: brew install portaudio (macOS) / apt install portaudio19-dev (Linux)
#   2. audio 组件: 在 SDK 仓库中位于 components/multimedia/audio
#      也支持独立开发时放在 stt 同级目录:
#        VOICE/
#        ├── stt/    # 本项目
#        └── audio/  # audio 组件
#      或通过 -DAUDIO_DIR=/path/to/audio 手动指定
#
# 使用方法:
#   cmake .. -DBUILD_STREAM_DEMO=ON
#
option(BUILD_STREAM_DEMO "Build stream demo and Python bindings (requires audio component)" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings (requires BUILD_STREAM_DEMO=ON)" ON)

# -------------------------------------------------------
# Find Dependencies
# -------------------------------------------------------
find_package(PkgConfig REQUIRED)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
    /opt/homebrew/include/onnxruntime
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
    /usr/local/lib
    /usr/lib
    /opt/homebrew/lib
)

# libsndfile
pkg_check_modules(SNDFILE REQUIRED sndfile)

# libcurl
pkg_check_modules(CURL REQUIRED libcurl)

# FFTW3
if(APPLE)
    find_library(STT_FFTW3F_LIB NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(STT_FFTW3_INCLUDE NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include)
    set(FFTW3_LIBRARIES ${STT_FFTW3F_LIB})
    set(FFTW3_INCLUDE_DIRS ${STT_FFTW3_INCLUDE})
    set(FFTW3_LIBRARY_DIRS /opt/homebrew/lib)
else()
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
endif()

# libsamplerate (optional)
pkg_check_modules(SAMPLERATE samplerate)

# -------------------------------------------------------
# Audio Component (optional, for stream demo)
# -------------------------------------------------------
# Search for audio component in multiple locations
if(NOT AUDIO_DIR)
    set(_AUDIO_SEARCH_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../multimedia/audio   # SDK repo: components/multimedia/audio
        ${CMAKE_CURRENT_SOURCE_DIR}/../audio                 # Standalone: sibling directory
    )
    foreach(_path ${_AUDIO_SEARCH_PATHS})
        if(EXISTS ${_path}/inc/audio_api.hpp)
            get_filename_component(AUDIO_DIR ${_path} ABSOLUTE)
            break()
        endif()
    endforeach()
endif()

set(AUDIO_SRC_DIR ${AUDIO_DIR}/src)
set(AUDIO_INC_DIR ${AUDIO_DIR}/inc)

# Check if audio component exists
if(BUILD_STREAM_DEMO)
    if(NOT AUDIO_DIR OR NOT EXISTS ${AUDIO_DIR}/inc/audio_api.hpp)
        message(FATAL_ERROR
            "BUILD_STREAM_DEMO=ON but audio component not found!\n"
            "Expected location: components/multimedia/audio (in SDK repo)\n"
            "Or as sibling directory: ../audio (standalone build)\n"
            "You can also specify it manually: cmake .. -DAUDIO_DIR=/path/to/audio\n"
            "\n"
            "Or disable stream demo:\n"
            "  cmake .. -DBUILD_STREAM_DEMO=OFF"
        )
    endif()

    # PortAudio (required for stream demo)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
    if(NOT PORTAUDIO_FOUND)
        message(FATAL_ERROR
            "BUILD_STREAM_DEMO=ON but PortAudio not found!\n"
            "Install PortAudio:\n"
            "  macOS: brew install portaudio\n"
            "  Linux: apt install portaudio19-dev"
        )
    endif()
endif()

# -------------------------------------------------------
# Resampler Library (requires audio component)
# -------------------------------------------------------
if(BUILD_STREAM_DEMO)
    set(RESAMPLER_SOURCES
        ${AUDIO_SRC_DIR}/resampler.cpp
        ${AUDIO_SRC_DIR}/resampler_rvv.cpp
    )

    add_library(stt_resampler STATIC ${RESAMPLER_SOURCES})

    target_include_directories(stt_resampler PUBLIC
        ${AUDIO_INC_DIR}
    )

    # Enable libsamplerate if available
    if(SAMPLERATE_FOUND)
        target_compile_definitions(stt_resampler PUBLIC USE_LIBSAMPLERATE)
        target_include_directories(stt_resampler PRIVATE ${SAMPLERATE_INCLUDE_DIRS})
        target_link_libraries(stt_resampler PUBLIC ${SAMPLERATE_LIBRARIES})
        target_link_directories(stt_resampler PUBLIC ${SAMPLERATE_LIBRARY_DIRS})
    endif()

    set_target_properties(stt_resampler PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# -------------------------------------------------------
# EvoAudio Library (for stream demo)
# -------------------------------------------------------
if(BUILD_STREAM_DEMO)
    add_library(stt_evo_audio STATIC
        ${AUDIO_SRC_DIR}/evo_audio.cpp
        ${AUDIO_SRC_DIR}/audio_stream.cpp
    )

    target_include_directories(stt_evo_audio PUBLIC
        ${AUDIO_INC_DIR}
        ${PORTAUDIO_INCLUDE_DIRS}
    )

    target_include_directories(stt_evo_audio PRIVATE
        ${AUDIO_INC_DIR}/internal
    )

    target_link_libraries(stt_evo_audio PUBLIC ${PORTAUDIO_LIBRARIES})
    target_link_directories(stt_evo_audio PUBLIC ${PORTAUDIO_LIBRARY_DIRS})

    set_target_properties(stt_evo_audio PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()

# -------------------------------------------------------
# SenseVoice Backend Library
# -------------------------------------------------------
set(SENSEVOICE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/sensevoice_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/feature_extractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/model_loader.cpp
)

add_library(sensevoice STATIC ${SENSEVOICE_SOURCES})

target_include_directories(sensevoice PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/internal
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(sensevoice
    PUBLIC ${ONNXRUNTIME_LIB}
    PUBLIC ${FFTW3_LIBRARIES}
    PRIVATE ${CURL_LIBRARIES}
    PRIVATE pthread
    PRIVATE m
)

target_link_directories(sensevoice
    PUBLIC ${FFTW3_LIBRARY_DIRS}
    PRIVATE ${CURL_LIBRARY_DIRS}
)

# -------------------------------------------------------
# STT Framework Library
# -------------------------------------------------------
set(STT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/sensevoice_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_adapter.cpp
)

add_library(stt STATIC ${STT_SOURCES})

target_include_directories(stt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/internal
)

# Conditionally add resampler include
if(BUILD_STREAM_DEMO)
    target_include_directories(stt PUBLIC ${AUDIO_INC_DIR})
    target_link_libraries(stt PUBLIC stt_resampler)
endif()

target_link_libraries(stt
    PUBLIC sensevoice
    PUBLIC ${SNDFILE_LIBRARIES}
    PRIVATE pthread
)

target_link_directories(stt PUBLIC ${SNDFILE_LIBRARY_DIRS})

set_target_properties(stt PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(sensevoice PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "STT Framework library enabled")

# -------------------------------------------------------
# STT Test Executable (always built)
# -------------------------------------------------------
add_executable(stt_test ${CMAKE_CURRENT_SOURCE_DIR}/examples/static_file_demo.cpp)

target_include_directories(stt_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/internal
)

target_link_libraries(stt_test
    PRIVATE stt
    PRIVATE ${SNDFILE_LIBRARIES}
    PRIVATE pthread
    PRIVATE m
)

target_link_directories(stt_test PRIVATE ${SNDFILE_LIBRARY_DIRS})

set_target_properties(stt_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "STT Test executable enabled")

# -------------------------------------------------------
# Stream Demo Executable (optional)
# -------------------------------------------------------
if(BUILD_STREAM_DEMO)
    add_executable(stream_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/stream_demo.cpp)

    target_include_directories(stream_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/inc/internal
        ${AUDIO_INC_DIR}
    )

    target_link_libraries(stream_demo
        PRIVATE stt
        PRIVATE stt_evo_audio
        PRIVATE stt_resampler
        PRIVATE ${SNDFILE_LIBRARIES}
        PRIVATE pthread
        PRIVATE m
    )

    target_link_directories(stream_demo PRIVATE ${SNDFILE_LIBRARY_DIRS})

    set_target_properties(stream_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    message(STATUS "Stream Demo executable enabled")
else()
    message(STATUS "Stream Demo disabled (BUILD_STREAM_DEMO=OFF)")
    message(STATUS "  To enable: cmake .. -DBUILD_STREAM_DEMO=ON")
    message(STATUS "  Requires: audio component + PortAudio")
endif()

# -------------------------------------------------------
# Python Bindings (optional)
# -------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    # Find Python first (respects -DPython3_EXECUTABLE=...)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)

    if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE} (${Python3_VERSION})")
    endif()

    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Try using Python's pybind11
        if(Python3_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE PYBIND11_RESULT
            )
            if(PYBIND11_RESULT EQUAL 0)
                list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
                find_package(pybind11 CONFIG QUIET)
            endif()
        endif()
    endif()

    if(pybind11_FOUND)
        pybind11_add_module(_evo_asr ${CMAKE_CURRENT_SOURCE_DIR}/python/asr_bindings.cpp)

        target_include_directories(_evo_asr PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/inc
            ${CMAKE_CURRENT_SOURCE_DIR}/inc/internal
        )

        target_link_libraries(_evo_asr PRIVATE stt)

        set_target_properties(_evo_asr PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
        )

        message(STATUS "Python bindings enabled (_evo_asr module for Python ${Python3_VERSION})")

        # Get Python site-packages path
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Install target for pip
        install(TARGETS _evo_asr LIBRARY DESTINATION .)

        # Custom target: stt-install-python (copy all files to virtualenv site-packages)
        add_custom_target(stt-install-python
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_SITE_PACKAGES}/evo_asr
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/evo_asr ${PYTHON_SITE_PACKAGES}/evo_asr
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_evo_asr> ${PYTHON_SITE_PACKAGES}/evo_asr/
            DEPENDS _evo_asr
            COMMENT "Installing evo_asr to ${PYTHON_SITE_PACKAGES}/evo_asr/"
        )
    else()
        message(STATUS "Python bindings disabled (pybind11 not found)")
        message(STATUS "  Install with: pip install pybind11 OR brew install pybind11")
    endif()
endif()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
message(STATUS "")
message(STATUS "=== STT Build Configuration ===")
message(STATUS "  BUILD_STREAM_DEMO:      ${BUILD_STREAM_DEMO}")
message(STATUS "  BUILD_PYTHON_BINDINGS:  ${BUILD_PYTHON_BINDINGS}")
message(STATUS "")
