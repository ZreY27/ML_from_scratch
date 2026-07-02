#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // OBLIGATOIRE pour convertir vector <-> list automatiquement
#include "../src/ImageLoader.hpp"
#include "../src/LinearModel.hpp"
#include "../src/MLP.hpp"
#include "../src/RBF.hpp"
#include "../src/SVM.h"

namespace py = pybind11;

// "ML_ESGI" sera le nom de l'import en Python
PYBIND11_MODULE(ML_ESGI, m) {
    m.doc() = "Bibliotheque ML C++ - Projet Annuel ESGI (Optimisee)"; // Documentation globale du module
    
    py::class_<LinearModel>(m, "LinearModel")
        .def(py::init<int, bool>(), py::arg("input_size"), py::arg("is_classification") = true, "Initialise le modele lineaire (Classification par defaut)")
        .def("train", &LinearModel::train,
             py::arg("inputs"),
             py::arg("labels"),
             py::arg("learning_rate"),
             py::arg("epochs"),
             "Entraine le modele sur le dataset et retourne l'historique des erreurs (loss)")
        .def("train_from_images", &LinearModel::train_from_images,
             py::arg("image_paths"),
             py::arg("labels"),
             py::arg("target_w"),
             py::arg("target_h"),
             py::arg("learning_rate"),
             py::arg("epochs"),
             "Entraine le modele directement depuis une liste de chemins d'images")
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
        .def(py::init<const std::vector<int>&, bool>(), py::arg("npl"), py::arg("is_classification") = true, "Initialise le MLP avec une architecture et un mode (Classification par defaut)")
        .def("train", &MLP::train,
             py::arg("dataset_inputs"),
             py::arg("dataset_expected_outputs"),
             py::arg("training_steps"),
             py::arg("learning_rate"),
             py::arg("decay") = 0.0,
             "Entraine le modele MLP via Backpropagation SGD avec Decay et retourne l'historique des erreurs (Loss)")
        .def("train_from_images", &MLP::train_from_images,
             py::arg("image_paths"),
             py::arg("expected_outputs"),
             py::arg("target_w"),
             py::arg("target_h"),
             py::arg("training_steps"),
             py::arg("learning_rate"),
             py::arg("decay") = 0.0,
             "Entraine le modele MLP directement depuis une liste d'images avec Decay et retourne la Loss")
        .def("predict", &MLP::predict,
             py::arg("inputs"),
             "Predit les valeurs pour une donnee et retourne la derniere couche")
        .def("save", &MLP::save,
             py::arg("filename"),
             "Sauvegarde les poids du modele")
        .def("load", &MLP::load,
             py::arg("filename"),
             "Charge les poids du modele");

    py::class_<RBF>(m, "RBF")
        .def(py::init<int, int, int, double, bool>(),
             py::arg("input_size"),
             py::arg("num_centers"),
             py::arg("output_size") = 1,
             py::arg("sigma") = 0.0,
             py::arg("is_classification") = true,
             "Initialise le reseau RBF (taille entree, nombre de centres, taille sortie, sigma, mode classif)")
        .def("train", &RBF::train,
             py::arg("inputs"),
             py::arg("labels"),
             py::arg("num_samples"),
             "Entraine le RBF : K-Means pour les centres puis moindres carres pour les poids")
        .def("train_from_images", &RBF::train_from_images,
             py::arg("image_paths"),
             py::arg("labels"),
             py::arg("target_w"),
             py::arg("target_h"),
             "Entraine le RBF directement depuis une liste de chemins d'images")
        .def("predict", &RBF::predict,
             py::arg("inputs"),
             "Predit la sortie pour une donnee (signe/argmax en classif, valeur brute en regression)")
        .def("predict_raw", &RBF::predict_raw,
             py::arg("inputs"),
             "Retourne les sorties brutes avant signe/argmax")
        .def("save", &RBF::save,
             py::arg("filename"),
             "Sauvegarde les centres, sigma et les poids du modele")
        .def("load", &RBF::load,
             py::arg("filename"),
             "Charge un modele RBF depuis un fichier");

    // SVM : Hinge loss (classification) ou epsilon-insensible / SVR (regression).
    // ⚠ Contrairement aux autres modeles, train() prend X en 2D (liste de listes), pas en 1D aplati.
    py::class_<SVM> svm(m, "SVM");

    py::enum_<SVM::Mode>(svm, "Mode")
        .value("CLASSIFICATION", SVM::CLASSIFICATION)
        .value("REGRESSION", SVM::REGRESSION)
        .export_values();

    svm.def(py::init<int, double, double, SVM::Mode>(),
            py::arg("input_size"),
            py::arg("lambda_reg") = 0.001,
            py::arg("epsilon") = 0.1,
            py::arg("mode") = SVM::CLASSIFICATION,
            "Initialise le SVM (input_size, lambda_reg = regularisation L2, epsilon = tube SVR, mode)")
       .def("train", &SVM::train,
            py::arg("X"),
            py::arg("Y"),
            py::arg("learning_rate"),
            py::arg("epochs"),
            "Entraine le SVM. X = liste de listes (2D), Y = labels -1/+1 (classif) ou valeurs (regression). La loss est dans loss_history.")
       .def("predict", &SVM::predict,
            py::arg("x"),
            "Predit la classe (-1.0 / +1.0) ou la valeur continue (regression)")
       .def("predict_raw", &SVM::predict_raw,
            py::arg("x"),
            "Retourne le score brut W.X + b (utile pour le multi-classe One-vs-Rest)")
       .def("save", &SVM::save,
            py::arg("filename"),
            "Sauvegarde mode, biais, lambda_reg, epsilon et les poids dans un fichier texte")
       .def("load", &SVM::load,
            py::arg("filename"),
            "Charge le modele depuis un fichier texte")
       .def_readonly("loss_history", &SVM::loss_history,
            "Historique de la loss par epoch (rempli pendant train)");

    m.def("load_and_resize_image", &load_and_resize_image,
          py::arg("filepath"),
          py::arg("target_w"),
          py::arg("target_h"),
          "Charge, redimensionne et normalise une image");
}
