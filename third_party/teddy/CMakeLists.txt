cmake_minimum_required(VERSION 3.10)
project(TeddySketch2Model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)

# Set root directory
get_filename_component(TEDDY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

# User-configurable dependency paths (with defaults)
set(EIGEN3_DIR "" CACHE PATH "Path to Eigen3 include directory (should contain Eigen/ subdirectory)")
set(LIBIGL_DIR "" CACHE PATH "Path to libigl include directory (should contain igl/ subdirectory)")
set(THIRD_PARTY_DIR "" CACHE PATH "Path to third_party directory containing image, ir3d-utils, miscutils, triangle, zip-mod, tinygltf, stb subdirectories")

# If THIRD_PARTY_DIR is not set, try to find it relative to this directory
if(NOT THIRD_PARTY_DIR)
    # Try common locations (priority: bundled dependencies first)
    set(POSSIBLE_THIRD_PARTY_DIRS
        "${TEDDY_ROOT}/third_party"                    # Bundled in third_party/teddy/third_party/
        "${TEDDY_ROOT}/../../../external/monster-mash/third_party"  # Legacy path (for backward compatibility)
        "${TEDDY_ROOT}/../../external/monster-mash/third_party"
    )
    foreach(POSSIBLE_DIR ${POSSIBLE_THIRD_PARTY_DIRS})
        if(EXISTS "${POSSIBLE_DIR}" AND EXISTS "${POSSIBLE_DIR}/image")
            set(THIRD_PARTY_DIR "${POSSIBLE_DIR}")
            break()
        endif()
    endforeach()
endif()

# Validate THIRD_PARTY_DIR
if(NOT THIRD_PARTY_DIR OR NOT EXISTS "${THIRD_PARTY_DIR}")
    message(FATAL_ERROR "THIRD_PARTY_DIR not found. Please set it to the directory containing image, ir3d-utils, etc.")
endif()

message(STATUS "Using third_party directory: ${THIRD_PARTY_DIR}")

# If EIGEN3_DIR is not set, try to find it
if(NOT EIGEN3_DIR)
    set(POSSIBLE_EIGEN3_DIRS
        "${TEDDY_ROOT}/third_party/eigen3"             # Bundled in third_party/teddy/third_party/eigen3
        "${THIRD_PARTY_DIR}/eigen3"                    # If THIRD_PARTY_DIR was found, check inside it
        "${TEDDY_ROOT}/../../../external/monster-mash/third_party/eigen3"
    )
    foreach(POSSIBLE_DIR ${POSSIBLE_EIGEN3_DIRS})
        if(EXISTS "${POSSIBLE_DIR}" AND EXISTS "${POSSIBLE_DIR}/Eigen")
            set(EIGEN3_DIR "${POSSIBLE_DIR}")
            break()
        endif()
    endforeach()
    
    # Try system Eigen3 (fallback)
    if(NOT EIGEN3_DIR)
        find_path(EIGEN3_DIR Eigen/Eigen
            PATHS
            /usr/include/eigen3
            /usr/local/include/eigen3
            /opt/homebrew/include/eigen3
        )
    endif()
endif()

if(NOT EIGEN3_DIR OR NOT EXISTS "${EIGEN3_DIR}")
    message(FATAL_ERROR "Eigen3 not found. Please set EIGEN3_DIR or install Eigen3.")
endif()

message(STATUS "Using Eigen3 from: ${EIGEN3_DIR}")

# If LIBIGL_DIR is not set, try to find it
if(NOT LIBIGL_DIR)
    set(POSSIBLE_LIBIGL_DIRS
        "${TEDDY_ROOT}/third_party/libigl/include"     # Bundled in third_party/teddy/third_party/libigl/include
        "${THIRD_PARTY_DIR}/libigl/include"            # If THIRD_PARTY_DIR was found, check inside it
        "${TEDDY_ROOT}/../../../external/monster-mash/third_party/libigl/include"
    )
    foreach(POSSIBLE_DIR ${POSSIBLE_LIBIGL_DIRS})
        if(EXISTS "${POSSIBLE_DIR}" AND EXISTS "${POSSIBLE_DIR}/igl")
            set(LIBIGL_DIR "${POSSIBLE_DIR}")
            break()
        endif()
    endforeach()
endif()

if(NOT LIBIGL_DIR OR NOT EXISTS "${LIBIGL_DIR}")
    message(FATAL_ERROR "libigl not found. Please set LIBIGL_DIR or install libigl.")
endif()

message(STATUS "Using libigl from: ${LIBIGL_DIR}")

# Set include directories
# IMPORTANT: stubs directory must be FIRST so SDL_opengles2.h stub is found before system headers
set(INCLUDEPATH
    ${TEDDY_ROOT}/stubs                          # SDL stubs (MUST BE FIRST)
    ${TEDDY_ROOT}/src                            # Core algorithm source
    ${TEDDY_ROOT}/src/common                     # Common headers
    ${TEDDY_ROOT}/eigen3-mod                     # Eigen modifications
    ${EIGEN3_DIR}                                # Eigen3 headers
    ${LIBIGL_DIR}                                # libigl headers
    ${THIRD_PARTY_DIR}                           # Third party root
    ${THIRD_PARTY_DIR}/image                     # Image library
    ${THIRD_PARTY_DIR}/ir3d-utils                # 3D utilities
    ${THIRD_PARTY_DIR}/miscutils                 # Misc utilities
    ${THIRD_PARTY_DIR}/triangle                  # Triangle library
    ${THIRD_PARTY_DIR}/zip-mod                   # ZIP library
    ${THIRD_PARTY_DIR}/tinygltf                  # TinyGLTF
    ${THIRD_PARTY_DIR}/stb                       # STB headers
)

# Source files
set(WRAPPER_CPP "${TEDDY_ROOT}/pybind11/wrapper.cpp")
set(WRAPPER_H "${TEDDY_ROOT}/pybind11/wrapper.h")
set(PREINCLUDE_H "${TEDDY_ROOT}/pybind11/preinclude.h")
set(SKETCH2MODEL_CPP "${TEDDY_ROOT}/src/sketch2model.cpp")

# Verify files exist
if(NOT EXISTS "${WRAPPER_CPP}")
    message(FATAL_ERROR "Cannot find wrapper.cpp at: ${WRAPPER_CPP}")
endif()

if(NOT EXISTS "${SKETCH2MODEL_CPP}")
    message(FATAL_ERROR "Cannot find sketch2model.cpp at: ${SKETCH2MODEL_CPP}")
endif()

# Source files from third_party (user must provide these)
set(SOURCES
    ${WRAPPER_CPP}
    ${SKETCH2MODEL_CPP}
    ${THIRD_PARTY_DIR}/ir3d-utils/regionToMesh.cpp
    ${THIRD_PARTY_DIR}/ir3d-utils/MeshBuilder.cpp
    ${THIRD_PARTY_DIR}/image/imageReadWrite.cpp
    ${THIRD_PARTY_DIR}/image/imageScale.cpp
    ${THIRD_PARTY_DIR}/image/imageUtils.cpp
    ${THIRD_PARTY_DIR}/miscutils/def3d.cpp
    ${THIRD_PARTY_DIR}/miscutils/mesh3d.cpp
    ${THIRD_PARTY_DIR}/miscutils/fsutils.cpp
    ${THIRD_PARTY_DIR}/triangle/triangle.c
    ${THIRD_PARTY_DIR}/zip-mod/zip.c
    ${THIRD_PARTY_DIR}/tinygltf/tinygltf.cpp
)

# Compile definitions
set(DEFINES
    IMAGE_READ_WRITE
    IMAGE_MINIZ_EXTERNAL
    IMAGE_SCALE
    TRILIBRARY
    NO_TIMER
    IGL_PARALLEL_FOR_FORCE_SERIAL
    EIGEN_SPARSESOLVERBASE_H
    EIGEN_DONT_INCLUDE_ITERATIVELINEARSOLVERS
    TINYGLTF_NO_STB_IMAGE
    TINYGLTF_NO_STB_IMAGE_WRITE
    TINYGLTF_NO_EXTERNAL_IMAGE
    TINYGLTF_USE_CPP14
    _USE_MATH_DEFINES
    NOMINMAX
)

# Platform-specific settings
if(WIN32)
    list(APPEND DEFINES NOMINMAX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable")
endif()

# Create Python module
pybind11_add_module(teddy_sketch2model ${SOURCES})

# Set target properties
set_target_properties(teddy_sketch2model PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(teddy_sketch2model PRIVATE ${INCLUDEPATH})

# Compile definitions
target_compile_definitions(teddy_sketch2model PRIVATE ${DEFINES})

# Windows-specific compile options
if(WIN32)
    target_compile_options(teddy_sketch2model PRIVATE
        /Zc:__cplusplus
        /Zc:preprocessor
        /permissive-
        /bigobj
        /std:c++17
        /wd4624
        /wd5026
        /wd5027
    )
    
    # Force include pre-include header for all source files
    if(EXISTS "${PREINCLUDE_H}")
        set_source_files_properties(${SKETCH2MODEL_CPP} PROPERTIES
            COMPILE_FLAGS "/FI\"${PREINCLUDE_H}\""
        )
        foreach(SOURCE_FILE ${SOURCES})
            if(SOURCE_FILE MATCHES "\\.cpp$" OR SOURCE_FILE MATCHES "\\.c$")
                get_filename_component(SOURCE_NAME ${SOURCE_FILE} NAME)
                if(NOT SOURCE_NAME STREQUAL "wrapper.cpp")
                    set_source_files_properties(${SOURCE_FILE} PROPERTIES
                        COMPILE_FLAGS "/FI\"${PREINCLUDE_H}\""
                    )
                endif()
            endif()
        endforeach()
    endif()
else()
    # Linux/Unix-specific: Force include pre-include header
    if(EXISTS "${PREINCLUDE_H}")
        get_filename_component(PREINCLUDE_H_ABS "${PREINCLUDE_H}" ABSOLUTE)
        message(STATUS "Using pre-include header: ${PREINCLUDE_H_ABS}")
        
        set_source_files_properties(${SKETCH2MODEL_CPP} PROPERTIES
            COMPILE_FLAGS "-include \"${PREINCLUDE_H_ABS}\""
        )
        foreach(SOURCE_FILE ${SOURCES})
            if(SOURCE_FILE MATCHES "\\.cpp$" OR SOURCE_FILE MATCHES "\\.c$")
                get_filename_component(SOURCE_NAME ${SOURCE_FILE} NAME)
                if(NOT SOURCE_NAME STREQUAL "wrapper.cpp")
                    get_source_file_property(CURRENT_FLAGS ${SOURCE_FILE} COMPILE_FLAGS)
                    if(CURRENT_FLAGS)
                        set_source_files_properties(${SOURCE_FILE} PROPERTIES
                            COMPILE_FLAGS "${CURRENT_FLAGS} -include \"${PREINCLUDE_H_ABS}\""
                        )
                    else()
                        set_source_files_properties(${SOURCE_FILE} PROPERTIES
                            COMPILE_FLAGS "-include \"${PREINCLUDE_H_ABS}\""
                        )
                    endif()
                endif()
            endif()
        endforeach()
    endif()
endif()

# Link libraries
target_link_libraries(teddy_sketch2model PRIVATE
    pybind11::module
    pybind11::lto
    pybind11::windows_extras
)

# Unix-specific linking
if(UNIX)
    target_link_libraries(teddy_sketch2model PRIVATE m)
endif()

# Print configuration
message(STATUS "Teddy Sketch2Model Configuration:")
message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
message(STATUS "  Python version: ${Python3_VERSION}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")
message(STATUS "  Eigen3 include: ${EIGEN3_DIR}")
message(STATUS "  libigl include: ${LIBIGL_DIR}")
message(STATUS "  Third party root: ${THIRD_PARTY_DIR}")
message(STATUS "  Teddy root: ${TEDDY_ROOT}")

