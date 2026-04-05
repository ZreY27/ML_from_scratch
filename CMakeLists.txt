cmake_minimum_required(VERSION 3.10) # <-- Cette ligne était manquante !
project(PROJET_ANNUEL) # <-- Celle-ci aussi !

set(CMAKE_CXX_STANDARD 14)

# 1. On trouve l'interpréteur Python du venv (grâce à votre config VS Code)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python utilisé : ${Python3_EXECUTABLE}")

# 2. On demande le chemin à pybind11 via une commande Python pure
# (Cette méthode est plus solide que l'argument --cmake qui a planté chez vous)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FOUND_EXIT_CODE
)

# Vérification si la commande a marché
if(NOT "${PYBIND11_FOUND_EXIT_CODE}" EQUAL "0")
    message(FATAL_ERROR "Impossible de trouver pybind11. Vérifiez 'pip install pybind11'.")
endif()

message(STATUS "Pybind11 trouvé ici : ${PYBIND11_CMAKE_DIR}")

# 3. On ajoute pybind11
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# 4. Vos fichiers sources
set(SOURCES
    src/LinearModel.cpp
    wrapper/binding.cpp
)

# 5. Création du module
pybind11_add_module(mon_module_cpp ${SOURCES})

# 6. Copie automatique vers le dossier scripts
add_custom_command(TARGET mon_module_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mon_module_cpp> ${CMAKE_SOURCE_DIR}/scripts/
)