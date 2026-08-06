cmake_minimum_required(VERSION 3.24)
project(era_translator LANGUAGES CXX)

# LibTorch 2.13 headers use designated initializers and bit-field default member
# initializers, which MSVC only accepts from C++20 on.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(MSVC)
    add_compile_options(/utf-8 /bigobj /EHsc)
endif()

include(FetchContent)

# ------------------------------------------------------------------ LibTorch
# Reuses an existing libtorch when CMAKE_PREFIX_PATH points at one, e.g. the
# copy shipped inside a PyTorch install:
#   -DCMAKE_PREFIX_PATH=<somewhere>/site-packages/torch
# Otherwise the official CPU build is downloaded (~250 MB, first configure only).
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    if(NOT WIN32)
        message(FATAL_ERROR
            "Automatic libtorch download is only wired up for Windows x64. "
            "Download libtorch for your platform and pass -DCMAKE_PREFIX_PATH=<libtorch>.")
    endif()
    message(STATUS "libtorch not on CMAKE_PREFIX_PATH - downloading CPU build (~250 MB)")
    FetchContent_Declare(libtorch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.13.0%2Bcpu.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR not-a-cmake-project)
    FetchContent_MakeAvailable(libtorch)
    list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
    find_package(Torch REQUIRED)
endif()

# -------------------------------------------------------------------- DuckDB
# Prebuilt library: DuckDB's own docs recommend the C API over the C++ API,
# which is documented as internal and unstable.
if(DEFINED DUCKDB_ROOT)
    set(duckdb_SOURCE_DIR "${DUCKDB_ROOT}")
else()
    if(NOT WIN32)
        message(FATAL_ERROR
            "Prebuilt DuckDB download is only wired up for Windows x64. "
            "Unpack libduckdb for your platform and pass -DDUCKDB_ROOT=<dir>.")
    endif()
    FetchContent_Declare(duckdb
        URL https://github.com/duckdb/duckdb/releases/download/v1.5.5/libduckdb-windows-amd64.zip
        URL_HASH SHA256=8375eb1fcf2212e8a0817950354815d4dde9dd383c2d9fa7b8975b71e278c1bd
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR not-a-cmake-project)
    FetchContent_MakeAvailable(duckdb)
endif()

add_library(duckdb_lib SHARED IMPORTED)
set_target_properties(duckdb_lib PROPERTIES
    IMPORTED_LOCATION "${duckdb_SOURCE_DIR}/duckdb.dll"
    IMPORTED_IMPLIB "${duckdb_SOURCE_DIR}/duckdb.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${duckdb_SOURCE_DIR}")

# ----------------------------------------------------------------- Matplot++
set(MATPLOTPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_INSTALLER OFF CACHE BOOL "" FORCE)
set(MATPLOTPP_BUILD_PACKAGE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
    GIT_TAG v1.2.2
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(matplotplusplus)

# Matplot++ shells out to gnuplot; without it the plot step cannot render.
find_program(GNUPLOT_EXECUTABLE gnuplot)
if(NOT GNUPLOT_EXECUTABLE)
    message(WARNING
        "gnuplot was not found on PATH. Matplot++ needs gnuplot 5.2.6+ at RUNTIME to "
        "render tsne_era.png; every other pipeline stage works without it.\n"
        "    Install with:  scoop install gnuplot   (or: choco install gnuplot)")
endif()

# ---------------------------------------------------------------- GoogleTest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------- GUI deps
# Everything here is permissively licensed, matching the rest of the project:
# Hello ImGui / Dear ImGui / ImPlot / ImPlot3D are MIT, GLFW is zlib. The GUI
# plots through ImPlot, so unlike the CLI it has no gnuplot requirement.
option(BUILD_GUI "Build the era_translator desktop cockpit" ON)
if(BUILD_GUI)
    # Hello ImGui supplies Dear ImGui (docking branch), GLFW, the OpenGL3
    # backend and the docking/window runner. It creates the `imgui` target.
    set(HELLOIMGUI_USE_GLFW3 ON CACHE BOOL "" FORCE)
    set(HELLOIMGUI_HAS_OPENGL3 ON CACHE BOOL "" FORCE)
    set(HELLOIMGUI_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
    set(HELLOIMGUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(HELLOIMGUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(hello_imgui
        GIT_REPOSITORY https://github.com/pthom/hello_imgui.git
        GIT_TAG v1.92.700
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(hello_imgui)

    # Dear ImGui indexes vertices with a 16-bit ImDrawIdx by default, which
    # overflows on a 23,516-point scatter. Widening it is the fix ImPlot3D's
    # docs call for, and PUBLIC propagates it everywhere imgui.h is included -
    # the definition must be identical in every translation unit.
    target_compile_definitions(imgui PUBLIC ImDrawIdx=unsigned\ int)

    # ImPlot and ImPlot3D ship sources with no CMake project of their own.
    FetchContent_Declare(implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG v1.0
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR not-a-cmake-project)
    FetchContent_Declare(implot3d
        GIT_REPOSITORY https://github.com/brenocq/implot3d.git
        GIT_TAG v0.4
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR not-a-cmake-project)
    FetchContent_MakeAvailable(implot implot3d)

    add_library(imgui_plots STATIC
        "${implot_SOURCE_DIR}/implot.cpp"
        "${implot_SOURCE_DIR}/implot_items.cpp"
        "${implot3d_SOURCE_DIR}/implot3d.cpp"
        "${implot3d_SOURCE_DIR}/implot3d_items.cpp"
        "${implot3d_SOURCE_DIR}/implot3d_meshes.cpp")
    target_include_directories(imgui_plots PUBLIC
        "${implot_SOURCE_DIR}" "${implot3d_SOURCE_DIR}")
    target_link_libraries(imgui_plots PUBLIC imgui)
endif()

# ------------------------------------------------------------------- Targets
add_library(era_translator_lib STATIC
    src/data.cpp
    src/evaluate.cpp
    src/features.cpp
    src/io.cpp
    src/model.cpp
    src/train.cpp)
target_include_directories(era_translator_lib PUBLIC src)
target_link_libraries(era_translator_lib PUBLIC "${TORCH_LIBRARIES}" duckdb_lib matplot)

add_executable(era_translator src/main.cpp)
target_link_libraries(era_translator PRIVATE era_translator_lib)

if(BUILD_GUI)
    set(GUI_SOURCES
        gui/main.cpp
        gui/app_state.cpp
        gui/panel_run.cpp
        gui/panel_metrics.cpp
        gui/panel_explore.cpp
        gui/panel_latent.cpp)
    # WIN32 means no console window, which in turn means the CRT wants WinMain.
    # Hello ImGui ships one that forwards to main(); its own add_app helper adds
    # this source, and a hand-rolled target has to do the same.
    if(WIN32)
        list(APPEND GUI_SOURCES
            "${hello_imgui_SOURCE_DIR}/hello_imgui_cmake/HelloImGui_WinMain.cpp")
    endif()
    add_executable(era_translator_gui WIN32 ${GUI_SOURCES})
    target_include_directories(era_translator_gui PRIVATE gui)
    target_link_libraries(era_translator_gui PRIVATE era_translator_lib hello_imgui imgui_plots)
endif()

enable_testing()
add_executable(era_translator_tests
    tests/test_evaluate.cpp
    tests/test_features.cpp
    tests/test_model.cpp
    tests/test_train.cpp)
target_link_libraries(era_translator_tests PRIVATE era_translator_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(era_translator_tests)

# Windows needs the torch and duckdb DLLs beside every executable that runs.
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    set(DLL_TARGETS era_translator era_translator_tests)
    if(BUILD_GUI)
        list(APPEND DLL_TARGETS era_translator_gui)
    endif()
    foreach(target ${DLL_TARGETS})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${duckdb_SOURCE_DIR}/duckdb.dll" $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS)
        if(TORCH_DLLS)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS} $<TARGET_FILE_DIR:${target}>
                COMMAND_EXPAND_LISTS)
        endif()
    endforeach()
endif()
