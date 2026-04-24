#include "LinearModel.hpp"
#include "ImageLoader.hpp"
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

std::vector<double> LinearModel::train(const std::vector<double>& inputs,
                                       const std::vector<double>& labels,
                                       double learning_rate, int epochs) {
    std::vector<double> loss_history;
    int input_size = weights.size();
    int num_samples = labels.size();

    for (int e = 0; e < epochs; e++) {
        // Barre de progression (issue de deploy)
        if (e % (epochs / 100 > 0 ? epochs / 100 : 1) == 0 || e == epochs - 1) {
            int progress = (int)((float)e / epochs * 100.0);
            std::cout << "\rTraining: [";
            for (int p = 0; p < 50; ++p) {
                if (p < progress / 2) std::cout << "=";
                else if (p == progress / 2) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << progress << "% " << std::flush;
        }

        if (mode == CLASSIFICATION) {
            int errors = 0;
            for (int i = 0; i < num_samples; i++) {
                // Calcul inline (optimisation de deploy) pour éviter l'allocation coûteuse
                double sum = bias;
                for (int j = 0; j < input_size; j++) {
                    sum += weights[j] * inputs[i * input_size + j];
                }
                double pred = (sum >= 0.0) ? 1.0 : -1.0;
                double error = labels[i] - pred;
                
                if (error != 0.0) {
                    errors++;
                    for (int j = 0; j < input_size; j++)
                        weights[j] += learning_rate * error * inputs[i * input_size + j];
                    bias += learning_rate * error;
                }
            }
            loss_history.push_back(static_cast<double>(errors) / num_samples);
        } else {
            double total_loss = 0.0;
            for (int i = 0; i < num_samples; i++) {
                // Calcul inline (optimisation de deploy)
                double sum = bias;
                for (int j = 0; j < input_size; j++) {
                    sum += weights[j] * inputs[i * input_size + j];
                }
                double pred = sum; // valeur brute (régression)
                double error = labels[i] - pred;
                total_loss += error * error;

                for (int j = 0; j < input_size; j++)
                    weights[j] += learning_rate * error * inputs[i * input_size + j];
                bias += learning_rate * error;
            }
            loss_history.push_back(total_loss / num_samples);
        }
    }
    std::cout << std::endl; // Nouvelle ligne propre à la fin de l'entraînement
    return loss_history;
}

std::vector<double> LinearModel::train_from_images(const std::vector<std::string>& image_paths, 
                                                   const std::vector<double>& labels, 
                                                   int target_w, int target_h, 
                                                   double learning_rate, int epochs) {
    std::vector<double> loss_history(epochs, 0.0);
    int input_size = weights.size();
    int num_samples = labels.size();

    for (int e = 0; e < epochs; e++) {
        if (mode == CLASSIFICATION) {
            int errors = 0;
            for (int i = 0; i < num_samples; i++) {
                std::vector<double> sample = load_and_resize_image(image_paths[i], target_w, target_h);
                double pred = predict(sample);
                double error = labels[i] - pred;

                if (error != 0.0) {
                    errors++;
                    for (int j = 0; j < input_size; j++)
                        weights[j] += learning_rate * error * sample[j];
                    bias += learning_rate * error;
                }
            }
            loss_history[e] = static_cast<double>(errors) / num_samples;
        } else {
            double total_loss = 0.0;
            for (int i = 0; i < num_samples; i++) {
                std::vector<double> sample = load_and_resize_image(image_paths[i], target_w, target_h);
                double pred = predict_raw(sample);
                double error = labels[i] - pred;
                total_loss += error * error;

                for (int j = 0; j < input_size; j++)
                    weights[j] += learning_rate * error * sample[j];
                bias += learning_rate * error;
            }
            loss_history[e] = total_loss / num_samples;
        }
    }
    return loss_history;
}

void LinearModel::save(const char* filename) {
    // TODO
}

void LinearModel::load(const char* filename) {
    // TODO
}