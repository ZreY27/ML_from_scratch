#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // OBLIGATOIRE pour convertir vector <-> list automatiquement
#include "../src/LinearModel.hpp"
#include "../src/MLP.hpp"
#include "../src/ImageLoader.hpp"

namespace py = pybind11;

// "ML_ESGI" sera le nom de l'import en Python
PYBIND11_MODULE(ML_ESGI, m) {
    m.doc() = "Bibliotheque ML C++ - Projet Annuel ESGI (Optimisee)"; // Documentation globale du module
    
    py::class_<LinearModel>(m, "LinearModel")
        .def(py::init<int>(), py::arg("input_size"), "Initialise le modele avec le nombre d'entrees (ex: pixels)")
        .def("train", &LinearModel::train,
             py::arg("inputs"),
             py::arg("labels"),
             py::arg("learning_rate"),
             py::arg("epochs"),
             "Entraine le modele sur le dataset et retourne l'historique des erreurs (loss)")
        .def("predict", &LinearModel::predict,
             py::arg("inputs"),
             "Predit la classe (1.0 ou -1.0) pour une image donnee")
        .def("predict_raw", &LinearModel::predict_raw,
             py::arg("inputs"),
             "Retourne le score brut de confiance (utile pour le multi-classe One-vs-Rest)")
        .def("save", &LinearModel::save,
             py::arg("filename"),
             "Sauvegarde les poids et le biais du modele dans un fichier texte")
        .def("load", &LinearModel::load,
             py::arg("filename"),
             "Charge les poids et le biais du modele depuis un fichier texte");

    py::class_<MLP>(m, "MLP")
        .def(py::init<const std::vector<int>&>(), py::arg("npl"), "Initialise le MLP avec une architecture (ex: [2, 3, 1])")
        .def("train", &MLP::train,
             py::arg("dataset_inputs"),
             py::arg("dataset_expected_outputs"),
             py::arg("training_steps"),
             py::arg("learning_rate"),
             py::arg("is_classification"),
             "Entraine le modele MLP via Backpropagation SGD")
        .def("predict", &MLP::predict,
             py::arg("inputs"),
             py::arg("is_classification"),
             "Predit les valeurs pour une donnee et retourne la derniere couche")
        .def("save", &MLP::save,
             py::arg("filename"),
             "Sauvegarde les poids du modele")
        .def("load", &MLP::load,
             py::arg("filename"),
             "Charge les poids du modele");

    m.def("load_and_resize_image", &load_and_resize_image,
          py::arg("filepath"),
          py::arg("target_w"),
          py::arg("target_h"),
          "Charge, redimensionne et normalise une image");
}
