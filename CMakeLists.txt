cmake_minimum_required(VERSION 3.10)
project(PROJET_ANNUEL)

set(CMAKE_CXX_STANDARD 17)
# 17 plutôt que 14 — quelques features C++ modernes en plus, aucun inconvénient

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python utilisé : ${Python3_EXECUTABLE}")

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

# ── Sources du module Python ──
set(SOURCES
        src/LinearModel.cpp
        wrapper/binding.cpp
)

# ── Module importable depuis Python ──
pybind11_add_module(mlmodels ${SOURCES})

# ── Copie automatique vers scripts/ après build ──
add_custom_command(TARGET mlmodels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:mlmodels>
        ${CMAKE_SOURCE_DIR}/scripts/
)

# ── Exécutable de test C++ pur ──
add_executable(test_linear
        src/main_test.cpp
        src/LinearModel.cpp
)