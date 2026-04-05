cmake_minimum_required(VERSION 3.10)
project(PROJET_ANNUEL)

set(CMAKE_CXX_STANDARD 14)

# 1. Détection automatique de Python selon l'OS
if(WIN32)
    find_program(Python3_EXECUTABLE NAMES python python3)
else()
    set(Python3_EXECUTABLE "/usr/bin/python3")
    if(NOT EXISTS "${Python3_EXECUTABLE}")
        find_program(Python3_EXECUTABLE NAMES python3.9 python3)
    endif()
endif()

# 2. Récupérer le répertoire CMake de pybind11 via Python, et échouer proprement si absent
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_FOUND_EXIT_CODE
)

if(NOT "${PYBIND11_FOUND_EXIT_CODE}" EQUAL "0")
    message(FATAL_ERROR "Impossible de trouver pybind11. Vérifiez 'pip3 install pybind11'.")
endif()

message(STATUS "Pybind11 trouvé ici : ${PYBIND11_CMAKE_DIR}")

# 3. Enregistrer pybind11 dans les chemins de recherche et le charger
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# 4. Fichiers sources : le modèle C++ et le binding pybind11 qui l'expose à Python
set(SOURCES
        src/LinearModel.cpp
        wrapper/binding.cpp
)

# 5. Compilation en module Python natif (.so / .pyd)
pybind11_add_module(mon_module_cpp ${SOURCES})

# 6. Copie automatique du module compilé dans scripts/ pour pouvoir l'importer directement
add_custom_command(TARGET mon_module_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mon_module_cpp> ${CMAKE_SOURCE_DIR}/scripts/
)