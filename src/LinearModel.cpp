// Auteur : Maxime Clément (voir LinearModel.hpp pour le détail de la partie individuelle)
#include "LinearModel.hpp"
#include "ImageLoader.hpp"
#include "ModelPath.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>   // std::setprecision (sauvegarde sans perte)
#include <stdexcept>
#include <filesystem>
#include <random>     // std::mt19937 (mélange des exemples à chaque epoch)
#include <algorithm>  // std::shuffle

LinearModel::LinearModel(int input_size, bool is_classification) : bias(0.0), is_classification(is_classification) {
    // Initialisation des poids : le cours (slide 65) autorise "random(-1,1) ou 0".
    // On prend un aléatoire petit (-0.01, 0.01) : même esprit, mais mieux adapté
    // à des entrées de grande dimension (3072 pixels normalisés entre 0 et 1).
    weights.resize(input_size);
    for (int i = 0; i < input_size; i++)
        weights[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 0.02 - 0.01;
}

double LinearModel::predict_raw(const std::vector<double>& inputs) const {
    // Garde-fou : une taille d'entrée incohérente (souvent une erreur côté Python)
    // provoquerait une lecture hors limites. pybind11 convertit cette exception
    // C++ en exception Python lisible au lieu d'un crash du process.
    if (inputs.size() != weights.size())
        throw std::invalid_argument("predict : l'entree a " + std::to_string(inputs.size()) +
                                    " valeurs mais le modele en attend " + std::to_string(weights.size()));

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
    int input_size = static_cast<int>(weights.size());
    int num_samples = static_cast<int>(labels.size());

    // Garde-fous : évite les lectures hors limites si les tableaux venus de
    // Python sont incohérents (X aplati doit contenir exactement
    // num_samples * input_size valeurs).
    if (num_samples == 0)
        throw std::invalid_argument("train : dataset vide (aucun label)");
    if (inputs.size() != static_cast<size_t>(num_samples) * static_cast<size_t>(input_size))
        throw std::invalid_argument("train : inputs contient " + std::to_string(inputs.size()) +
                                    " valeurs, attendu " + std::to_string(num_samples) + " x " +
                                    std::to_string(input_size));

    // Ordre de passage des exemples, re-mélangé à chaque epoch.
    // C'est l'esprit du cours (slide 65 : "prendre un exemple ... au hasard") :
    // un ordre fixe peut créer des cycles de mises à jour qui se compensent.
    // Graine fixe (42) -> résultats reproductibles d'une exécution à l'autre.
    std::vector<int> order(num_samples);
    for (int i = 0; i < num_samples; i++) order[i] = i;
    std::mt19937 rng(42);

    for (int e = 0; e < epochs; e++) {
        std::shuffle(order.begin(), order.end(), rng);
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
            for (int i : order) {
                // Somme pondérée calculée inline (évite une allocation de vecteur par exemple)
                double sum = bias;
                for (int j = 0; j < input_size; j++) {
                    sum += weights[j] * inputs[i * input_size + j];
                }
                double pred = (sum >= 0.0) ? 1.0 : -1.0;  // g(Xk) = Sign(W.X + b)
                double error = labels[i] - pred;          // (Yk - g(Xk)) : vaut 0, +2 ou -2 en labels -1/+1

                // Règle de Rosenblatt (slide 65) : W <- W + alpha * (Yk - g(Xk)) * Xk
                // Si l'exemple est bien classé, (Yk - g(Xk)) = 0 : aucune mise à jour.
                // Ex concret : Y=+1 mais g(X)=-1 -> error=+2 -> on AJOUTE 2*alpha*X aux
                // poids : le score de cette image remonte, l'hyperplan pivote vers elle.
                // (Y=-1 mais g(X)=+1 -> error=-2 -> on soustrait : le score descend.)
                if (error != 0.0) {
                    errors++;
                    for (int j = 0; j < input_size; j++)
                        weights[j] += learning_rate * error * inputs[i * input_size + j];
                    // Le biais correspond au poids w0 associé à l'entrée fictive x0 = 1
                    // (cf. slide 63 : "en prenant soin d'ajouter le biais x0 = 1").
                    bias += learning_rate * error;
                }
            }
            // Loss de classification = proportion d'exemples mal classés cette epoch (E_in)
            loss_history.push_back(static_cast<double>(errors) / num_samples);
        } else {
            // Régression : on minimise l'erreur quadratique moyenne par descente de
            // gradient (règle delta). Le cours (slide 66) présente la pseudo-inverse
            // W = (X^T X)^-1 X^T Y qui donne la solution exacte "en un coup" ; on a
            // choisi la version itérative car elle évite d'inverser une matrice
            // 3073x3073 pour des images, sans bibliothèque d'algèbre linéaire externe.
            double total_loss = 0.0;
            for (int i : order) {
                double sum = bias;
                for (int j = 0; j < input_size; j++) {
                    sum += weights[j] * inputs[i * input_size + j];
                }
                double pred = sum; // sortie brute (pas de fonction signe en régression)
                double error = labels[i] - pred;
                total_loss += error * error;

                // Gradient de (y - W.X)^2 par rapport à W : -2 * error * X
                // (le facteur 2 est absorbé dans le learning_rate)
                for (int j = 0; j < input_size; j++)
                    weights[j] += learning_rate * error * inputs[i * input_size + j];
                bias += learning_rate * error;
            }
            // Loss de régression = erreur quadratique moyenne (MSE) de l'epoch
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
    // Garde-fou : un label par image, sinon l'accès labels[i] déborde.
    if (image_paths.size() != labels.size())
        throw std::invalid_argument("train_from_images : " + std::to_string(image_paths.size()) +
                                    " images mais " + std::to_string(labels.size()) + " labels");

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
    // 17 chiffres significatifs : un double est restitue a l\'identique au load()
    // (la precision par defaut de C++ est de 6 chiffres -> poids legerement degrades)
    file << std::setprecision(17);
    
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