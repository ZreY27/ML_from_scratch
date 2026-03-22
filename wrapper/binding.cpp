#include <pybind11/pybind11.h>
// pybind11 = bibliothèque qui fait le pont entre C++ et Python
// elle génère un .so que Python peut importer directement

#include <pybind11/stl.h>
// stl.h = OBLIGATOIRE
// sans ça, std::vector<double> ne se convertit pas
// automatiquement en list Python et vice versa

#include "../src/LinearModel.h"
// on inclut notre classe C++

namespace py = pybind11;
// raccourci pour éviter d'écrire pybind11:: partout
// exactement comme "using namespace std" mais plus propre

// ─────────────────────────────────────────
// PYBIND11_MODULE = point d'entrée du module Python
// "mlmodels" = nom du module côté Python
//   → import mlmodels
// "m" = l'objet module sur lequel on enregistre tout
// ─────────────────────────────────────────
PYBIND11_MODULE(mlmodels, m) {
    m.doc() = "Bibliotheque ML C++ - Projet Annuel ESGI";// description affichée avec help(mlmodels) en Python
    py::class_ <LinearModel>(m, "LinearModel")
        .def(py::init<int>())// le constructeur prend un int en paramètre
        // Python : model = mlmodels.LinearModel(2)
        .def("predict", &LinearModel::predict,//méthode predict
             py::arg("x"),
             "Predit -1.0 ou +1.0 pour un exemple x")
                // "predict" = nom de la méthode côté Python
                 // &LinearModel::predict = pointeur vers la méthode C++
                 // py::arg("x") = nom de l'argument côté Python)

        .def("train", &LinearModel::train,//méthode train
            py::arg("X"),
            py::arg("Y"),
            py::arg("learning_rate"),
            py::arg("epochs"),
            "Entraine le modèle sur le dataset X avec les labes Y")
            // X et Y sont des list Python → convertis en vector C++ automatiquement
            // grâce au #include <pybind11/stl.h>
            .def("save", &LinearModel::save,
                py::arg("filename"),
                "Sauvegarde le modèle dans un fichier")
        .def_readonly("weights", &LinearModel::weights)
        .def_readonly("bias", &LinearModel::bias)
        .def_readonly("loss_history", &LinearModel::loss_history);
}