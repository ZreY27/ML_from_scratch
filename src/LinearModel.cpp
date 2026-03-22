#include "LinearModel.hpp"
#include <iostream>
#include <random>

LinearModel::LinearModel(int input_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < input_size; ++i) {
        weights.push_back(dis(gen));
    }
    bias = dis(gen);
}

double LinearModel::predict(const std::vector<double>& inputs) const {
    double sum = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * inputs[i];
    }
    // Fonction d'activation (signe) pour la classification binaire
    return (sum >= 0.0) ? 1.0 : -1.0;
}

void LinearModel::train(const std::vector<double>& inputs, const std::vector<double>& labels, double learning_rate, int epochs) {
    int input_size = weights.size();
    int num_samples = labels.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_samples; ++i) {
            // Forward pass (inline pour éviter de copier un vecteur à chaque itération - gain de performance énorme)
            double sum = bias;
            for (int j = 0; j < input_size; ++j) {
                sum += weights[j] * inputs[i * input_size + j];
            }
            double prediction = (sum >= 0.0) ? 1.0 : -1.0;

            double error = labels[i] - prediction;

            // Backward pass (Règle d'apprentissage du Perceptron)
            if (error != 0.0) {
                for (int j = 0; j < input_size; ++j) {
                    weights[j] += learning_rate * error * inputs[i * input_size + j];
                }
                bias += learning_rate * error;
            }
        }
    }
}

void LinearModel::save(const char* filename) {

}