cmake_minimum_required(VERSION 3.14)
project(PROJET_ANNUEL)

# On passe en C++17 car les versions récentes de Pybind11 et Eigen en tirent de meilleures performances
set(CMAKE_CXX_STANDARD 17)

# Demande à CMake de générer le fichier de configuration pour VS Code (IntelliSense)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 1. Python et Pybind11 ---
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python utilisé : ${Python3_EXECUTABLE}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FOUND_EXIT_CODE
)

if(NOT "${PYBIND11_FOUND_EXIT_CODE}" EQUAL "0")
    message(FATAL_ERROR "Impossible de trouver pybind11. Avez-vous fait 'pip install pybind11' dans votre environnement Python ?")
endif()

message(STATUS "Pybind11 trouvé ici : ${PYBIND11_CMAKE_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# --- 2. Eigen ---
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)

# --- 3. STB (Pour lire les images en C++) ---
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# --- 4. Compilation du module ---
set(SOURCES
    src/LinearModel.cpp
    src/MLP.cpp
    src/ImageLoader.cpp
    wrapper/binding.cpp
)

pybind11_add_module(mon_module_cpp ${SOURCES})

target_link_libraries(mon_module_cpp PRIVATE Eigen3::Eigen)
target_include_directories(mon_module_cpp PRIVATE ${stb_SOURCE_DIR})

# Copier le module généré dans le dossier scripts/
add_custom_command(TARGET mon_module_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mon_module_cpp> ${CMAKE_SOURCE_DIR}/scripts/
)
