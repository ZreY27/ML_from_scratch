#include "LinearModel.hpp"
#include "ImageLoader.hpp"
#include "ModelPath.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

LinearModel::LinearModel(int input_size, bool is_classification) : bias(0.0), is_classification(is_classification) {
    // on stocke le mode passé en paramètre
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
    if (is_classification)
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

        if (is_classification) {
            int errors = 0;
            for (int i = 0; i < num_samples; i++) {
                // Calcul inline (optimisation de deploy) pour éviter l'allocation coûteuse
                double sum = bias;
                for (int j = 0; j < input_size; j++) {
                    sum += weights[j] * inputs[i * input_size + j];
                }
                double pred = (sum >= 0.0) ? 1.0 : -1.0;
                double error = labels[i] - pred;
                
                // Règle de Rosenblatt : Mise à jour des poids uniquement en cas d'erreur
                if (error != 0.0) {
                    errors++;
                    // Formule de Rosenblatt : W = W + learning_rate * (Y_attendu - Y_predit) * X
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
    std::vector<double> flattened_inputs;
    std::vector<double> valid_labels;

    for (size_t i = 0; i < image_paths.size(); i++) {
        try {
            std::vector<double> sample = load_and_resize_image(image_paths[i], target_w, target_h);
            flattened_inputs.insert(flattened_inputs.end(), sample.begin(), sample.end());
            valid_labels.push_back(labels[i]); // on ne garde le label que si l'image a bien chargé
        } catch (const std::exception& e) {
            std::cerr << "  [C++] Erreur ignorée pour l'image " << image_paths[i] << " : " << e.what() << "\n";
        }
    }

    // Entraînement (images et labels restent alignés : une image ignorée l'est des deux côtés)
    return train(flattened_inputs, valid_labels, learning_rate, epochs);
}

void LinearModel::save(const char* filename) {
    std::string path = resolve_model_path(filename, true);

    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Erreur save LinearModel");
    
    // On sauvegarde le mode, le biais, puis les poids
    file << is_classification << "\n";
    file << bias << "\n";
    for (size_t i = 0; i < weights.size(); i++) {
        file << weights[i] << (i == weights.size() - 1 ? "" : " ");
    }
    file << "\n";
    file.close();
}

void LinearModel::load(const char* filename) {
    std::string path = resolve_model_path(filename);

    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error(std::string("Erreur : Impossible de charger le fichier ") + path);

    bool mode_val;
    if (file >> mode_val) {
        is_classification = mode_val;
    }
    file >> bias;
    
    weights.clear();
    double w;
    while (file >> w) {
        weights.push_back(w);
    }
    file.close();
}