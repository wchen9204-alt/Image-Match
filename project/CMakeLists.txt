cmake_minimum_required(VERSION 3.16)
project(ImageRegistrationFramework
    VERSION 0.1.0
    DESCRIPTION "Feature-based Image Registration Framework"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Compiler and language settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/W3 /utf-8 /EHsc /permissive-)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# OpenCV configuration
# ---------------------------------------------------------------------------
set(OPENCV_INSTALL_ROOT "D:/Opcv/opencv-contrib-mingw-install" CACHE PATH "OpenCV installation root")
list(APPEND CMAKE_PREFIX_PATH "${OPENCV_INSTALL_ROOT}")

find_package(OpenCV CONFIG REQUIRED
    COMPONENTS
        core
        imgproc
        highgui
        features2d
        xfeatures2d
        imgcodecs
        flann
        calib3d
)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ---------------------------------------------------------------------------
# yaml-cpp configuration
# ---------------------------------------------------------------------------
set(YAML_CPP_ROOT "D:/yaml-cpp-install" CACHE PATH "yaml-cpp installation root")
list(APPEND CMAKE_PREFIX_PATH "${YAML_CPP_ROOT}")

find_package(yaml-cpp CONFIG REQUIRED)
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET YAML::yaml-cpp)
    set(YAML_CPP_TARGET YAML::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp imported target not found.")
endif()
message(STATUS "yaml-cpp target: ${YAML_CPP_TARGET}")

# ---------------------------------------------------------------------------
# Core library: ir_core
# ---------------------------------------------------------------------------
file(GLOB_RECURSE IR_CORE_SOURCES
    CONFIGURE_DEPENDS
    "src/*.cpp"
)

add_library(ir_core STATIC ${IR_CORE_SOURCES})

target_include_directories(ir_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ir_core
    PUBLIC
        ${OpenCV_LIBS}
        ${YAML_CPP_TARGET}
)

target_compile_features(ir_core PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Application executable
# ---------------------------------------------------------------------------
add_executable(registration_app
    main.cpp
    apps/registration_app.cpp
)

target_link_libraries(registration_app
    PRIVATE
        ir_core
)

target_include_directories(registration_app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/apps
)

# ---------------------------------------------------------------------------
# Output and install-like copy steps
# ---------------------------------------------------------------------------
set_target_properties(registration_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(WIN32)
    file(GLOB OPENCV_RUNTIME_DLLS
        "${OPENCV_INSTALL_ROOT}/x64/mingw/bin/*.dll"
    )

    if(OPENCV_RUNTIME_DLLS)
        add_custom_command(TARGET registration_app POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:registration_app>
        )
    endif()
endif()

add_custom_command(TARGET registration_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/configs
        ${CMAKE_BINARY_DIR}/bin/configs
)
