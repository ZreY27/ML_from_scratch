#include "SVM.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

SVM::SVM(int input_size, double C, double epsilon, Mode mode) : bias(0.0), C(C), epsilon(epsilon), mode(mode){
    weights.resize(input_size);
    for (int i = 0; i < input_size; i++)
        weights[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 0.02 - 0.01;
}

double SVM::predict_raw(const std::vector<double> &x) const {
    double sum = bias;
    for (int i = 0; i < static_cast<int>(weights.size()); i++)
        sum += weights[i] * x[i];
    return sum;
}

double SVM::predict(const std::vector<double> &x) const {
    double raw = predict_raw(x);
    if (mode == CLASSIFICATION) {
        return predict_raw(x) >= 0.0 ? 1.0 : -1.0;
    }
    else {
        return raw;
    }

}

void SVM::train(const std::vector<std::vector<double> > &X, const std::vector<double> &Y, double learning_rate, int epochs) {
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
                    // Bien classifie : regularisation seulement
                    // w = w - lr * 2 * C * w, C c'est l'hyperparamètre de régularisation
                    for (int j = 0; j < static_cast<int>(weights.size()); j++) {
                        weights[j] -= learning_rate * 2 * C * weights[j];
                    }
                } else {
                    // Mal classifie ou dans la marge : regularisation + correction
                    // w = w - lr * (2*C*w - y*x)
                    // b = b + lr * y
                    for (int j = 0; j < static_cast<int>(weights.size()); j++) {
                        weights[j] -= learning_rate * (2 * C * weights[j] - Y[i] * X[i][j]);
                        bias += learning_rate * Y[i];
                    }
                }

                    }
                }
            }
            else {
                //TODO régréssion
            }
        }
    }
}

void SVM::save(const char *filename) {
    //TODO
}

void SVM::load(const char *filename) {
    //TODO
}

