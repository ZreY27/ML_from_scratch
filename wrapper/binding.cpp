#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // OBLIGATOIRE pour convertir vector <-> list automatiquement
#include "../src/LinearModel.hpp"
#include "../src/ImageLoader.hpp"

namespace py = pybind11;

// "ML_ESGI" sera le nom de l'import en Python
PYBIND11_MODULE(ML_ESGI, m) {
    
    py::class_<LinearModel>(m, "LinearModel")
        .def(py::init<int>())
        .def("train", &LinearModel::train)
        .def("predict", &LinearModel::predict)
        .def("predict_raw", &LinearModel::predict_raw)
        .def("save", &LinearModel::save)
        .def("load", &LinearModel::load);

    m.def("load_and_resize_image", &load_and_resize_image, "Charge, redimensionne et normalise une image");
}