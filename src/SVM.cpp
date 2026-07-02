// Auteur : Maxime Clément (voir SVM.h pour le détail de la partie individuelle)
#include "SVM.h"
#include "ModelPath.hpp"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

SVM::SVM(int input_size, double lambda_reg, double epsilon, Mode mode) : bias(0.0), lambda_reg(lambda_reg), epsilon(epsilon), mode(mode){
    weights.resize(input_size);
    for (int i = 0; i < input_size; i++)
        weights[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 0.02 - 0.01;
}

double SVM::predict_raw(const std::vector<double> &x) const {
    // Garde-fou : une entrée de mauvaise taille provoquerait une lecture hors
    // limites ; pybind11 convertit l'exception en erreur Python lisible.
    if (x.size() != weights.size())
        throw std::invalid_argument("predict : l'entree a " + std::to_string(x.size()) +
                                    " valeurs mais le modele en attend " + std::to_string(weights.size()));

    double sum = bias;
    for (int i = 0; i < static_cast<int>(weights.size()); i++)
        sum += weights[i] * x[i];
    return sum;
}

double SVM::predict(const std::vector<double> &x) const {
    double raw = predict_raw(x);
    if (mode == CLASSIFICATION) {
        return raw >= 0.0 ? 1.0 : -1.0;
    }
    else {
        return raw;
    }

}

void SVM::train(const std::vector<std::vector<double> > &X, const std::vector<double> &Y, double learning_rate, int epochs) {
    // Garde-fous : tailles cohérentes avant d'entraîner (sinon accès hors limites)
    if (X.empty())
        throw std::invalid_argument("train : dataset vide");
    if (X.size() != Y.size())
        throw std::invalid_argument("train : " + std::to_string(X.size()) + " exemples mais " +
                                    std::to_string(Y.size()) + " labels");
    for (size_t i = 0; i < X.size(); i++)
        if (X[i].size() != weights.size())
            throw std::invalid_argument("train : l'exemple " + std::to_string(i) + " a " +
                                        std::to_string(X[i].size()) + " valeurs, attendu " +
                                        std::to_string(weights.size()));

    // Un nouvel appel à train() repart d'un historique vide (sinon les courbes
    // de plusieurs entraînements successifs se mélangeraient).
    loss_history.clear();

    for (int e = 0; e < epochs; e++) {
        double total_loss = 0.0;
        for (int i = 0; i < static_cast<int>(X.size()); i++) {
            if (mode == CLASSIFICATION) {
                // Hinge Loss
                // margin = y * (W.X + b)
                // si margin >= 1 : bien classifie avec marge suffisante
                // si margin <  1 : mal classifie ou dans la marge
                double score = predict_raw(X[i]);
                double margin = score * Y[i];
                double hinge = std::max(0.0, 1 - margin);
                total_loss += hinge;
                if (margin >= 1.0) {
                    // Bien classifie avec marge : regularisation seulement
                    // w = w - lr * 2 * lambda * w  (gradient de lambda*||w||^2)
                    for (int j = 0; j < static_cast<int>(weights.size()); j++) {
                        weights[j] -= learning_rate * 2 * lambda_reg * weights[j];
                    }
                } else {
                    // Mal classifie ou dans la marge : regularisation + correction
                    // w = w - lr * (2*lambda*w - y*x)   (sous-gradient de la hinge : -y*x)
                    // b = b + lr * y
                    for (int j = 0; j < static_cast<int>(weights.size()); j++) {
                        weights[j] -= learning_rate * (2 * lambda_reg * weights[j] - Y[i] * X[i][j]);
                    }
                    bias += learning_rate * Y[i];
                }
            }
            else {
                // Epsilon-insensitive Loss (SVR)
                // residual = prediction - valeur_reelle
                // si |residual| <= epsilon : dans le tube, pas de correction
                // si |residual| >  epsilon : hors du tube, on corrige
                double pred = predict_raw(X[i]);
                double residual = pred - Y[i];
                double abs_residual = std::abs(residual);

                // Epsilon-insensitive loss pour cet exemple
                double loss = std::max(0.0, abs_residual - epsilon);
                total_loss += loss;

                if (abs_residual <= epsilon) {
                    // Dans le tube : regularisation seulement
                    for (int j = 0; j < static_cast<int>(weights.size()); j++){
                        weights[j] -= learning_rate * 2.0 * lambda_reg * weights[j];
                    }
                }
                else {
                    // Hors du tube : regularisation + correction
                    // signe(residual) indique la direction de correction
                    // si residual > 0 : on predit trop haut, on diminue
                    // si residual < 0 : on predit trop bas, on augmente
                    double sign = (residual > 0) ? 1.0 : -1.0;
                    for (int j = 0; j < static_cast<int>(weights.size()); j++){
                        weights[j] -= learning_rate * (2.0 * lambda_reg * weights[j] + sign * X[i][j]);
                    }
                    bias -= learning_rate * sign;
                }
            }
        }
        total_loss /= static_cast<double>(X.size());
        loss_history.push_back(total_loss);
    }
}

void SVM::save(const char *filename) {
    std::string path = resolve_model_path(filename, true);

    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Erreur save SVM");

    file << mode << "\n";
    file << bias << "\n";
    file << lambda_reg << "\n";
    file << epsilon << "\n";
    for (size_t i = 0; i < weights.size(); i++) {
        file << weights[i] << (i == weights.size() - 1 ? "" : " ");
    }
    file << "\n";
    file.close();
}

void SVM::load(const char *filename) {
    std::string path = resolve_model_path(filename);

    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Erreur load SVM : " + path);

    int m;
    if (file >> m) mode = static_cast<Mode>(m);
    file >> bias >> lambda_reg >> epsilon;

    weights.clear();
    double w;
    while (file >> w) {
        weights.push_back(w);
    }
    file.close();
}
