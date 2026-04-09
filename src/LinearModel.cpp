#include "LinearModel.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

LinearModel::LinearModel(int input_size, Mode mode) : bias(0.0), mode(mode) {
    // mode(mode) = on stocke le mode passé en paramètre
    weights.resize(input_size);
    for (int i = 0; i < input_size; i++)
        weights[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 0.02 - 0.01;
}

double LinearModel::predict_raw(const std::vector<double>& inputs) const {
    // valeur brute W·X + b — utilisée en interne par predict() et train()
    double sum = bias;
    for (int i = 0; i < static_cast<int>(weights.size()); i++)
        sum += weights[i] * inputs[i];
    return sum;
}

double LinearModel::predict(const std::vector<double>& inputs) const {
    double raw = predict_raw(inputs);
    if (mode == CLASSIFICATION)
        return raw >= 0.0 ? 1.0 : -1.0;  // signe -> -1 ou +1
    else
        return raw;  // régression -> valeur continue directe
}

void LinearModel::train(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& Y,
                        double learning_rate, int epochs) {
    for (int e = 0; e < epochs; e++) {

        if (mode == CLASSIFICATION) {
            // ── Règle de Rosenblatt ──────────────────────────────
            // on ne corrige que si la prédiction est fausse
            int errors = 0;
            for (int i = 0; i < static_cast<int>(X.size()); i++) {
                double pred = predict(X[i]);
                double error = Y[i] - pred;
                if (error != 0.0) {
                    errors++;
                    for (int j = 0; j < static_cast<int>(weights.size()); j++)
                        weights[j] += learning_rate * error * X[i][j];
                    bias += learning_rate * error;
                }
            }
            loss_history.push_back(static_cast<double>(errors) / X.size());

        } else {
            // ── Descente de gradient MSE ─────────────────────────
            // on corrige toujours, proportionnellement à l'erreur
            // MSE = Mean Squared Error = moyenne des (pred - y)²
            // dérivée MSE par rapport à w[j] = -2 * error * x[j]
            // → w[j] += lr * error * x[j]  (même formule qu'en classif !)
            // la différence : error = Y[i] - predict_raw() (pas de signe)
            double total_loss = 0.0;
            for (int i = 0; i < static_cast<int>(X.size()); i++) {
                double pred = predict_raw(X[i]);
                // on utilise predict_raw et pas predict
                // car en régression predict() = predict_raw() de toute façon
                // mais c'est plus explicite comme ça
                double error = Y[i] - pred;
                total_loss += error * error;  // accumule l'erreur quadratique
                for (int j = 0; j < static_cast<int>(weights.size()); j++)
                    weights[j] += learning_rate * error * X[i][j];
                bias += learning_rate * error;
            }
            // loss = MSE de cette epoch
            loss_history.push_back(total_loss / X.size());
        }
    }
}

void LinearModel::save(const char* filename) {
    // TODO
}

void LinearModel::load(const char* filename) {
    // TODO
}