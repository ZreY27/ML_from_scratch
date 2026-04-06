cmake_minimum_required(VERSION 3.14)
project(PROJET_ANNUEL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 1. Python : détection cross-platform (Windows vs Mac/Linux) ---
if(WIN32)
    find_program(Python3_EXECUTABLE NAMES python python3)
else()
    set(Python3_EXECUTABLE "/usr/bin/python3")
    if(NOT EXISTS "${Python3_EXECUTABLE}")
        find_program(Python3_EXECUTABLE NAMES python3.9 python3)
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python utilisé : ${Python3_EXECUTABLE}")

# --- 2. Pybind11 ---
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_FOUND_EXIT_CODE
)

if(NOT "${PYBIND11_FOUND_EXIT_CODE}" EQUAL "0")
    message(FATAL_ERROR "Impossible de trouver pybind11. Vérifiez 'pip install pybind11'.")
endif()

message(STATUS "Pybind11 trouvé ici : ${PYBIND11_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# --- 3. Eigen ---
include(FetchContent)
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# --- 4. STB (lecture d'images en C++) ---
FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# --- 5. Compilation du module ---
set(SOURCES
        src/LinearModel.cpp
        src/MLP.cpp
        src/ImageLoader.cpp
        wrapper/binding.cpp
)

pybind11_add_module(ML_ESGI ${SOURCES})

target_link_libraries(ML_ESGI PRIVATE Eigen3::Eigen)
target_include_directories(ML_ESGI PRIVATE ${stb_SOURCE_DIR})

# Copie automatique du module compilé à la racine (à côté de votre train_test.py) après le build
add_custom_command(TARGET ML_ESGI POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ML_ESGI> ${CMAKE_SOURCE_DIR}/
)
