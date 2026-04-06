#include "LinearModel.hpp"
#include <iostream>
#include <random>
#include <fstream>
#include <stdexcept>
#include "ImageLoader.hpp"

// Constructeur : Initialisation du modèle.
// Les poids et le biais sont initialisés avec des valeurs aléatoires comprises entre -1.0 et 1.0.
// Cela permet de "briser la symétrie" au départ (si tout était à 0, le modèle aurait du mal à apprendre).
LinearModel::LinearModel(int input_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < input_size; ++i) {
        weights.push_back(dis(gen));
    }
    bias = dis(gen);
}

// Prédiction (Forward pass) pour UNE seule image.
// Reçoit les pixels de l'image (aplatie en 1D), les multiplie par les poids appris, et ajoute le biais.
double LinearModel::predict(const std::vector<double>& inputs) const {
    double sum = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * inputs[i];
    }
    // Fonction d'activation "Signe" (Seuil à 0.0) pour la classification binaire.
    return (sum >= 0.0) ? 1.0 : -1.0;
}

// Entraînement du modèle (Algorithme du Perceptron de Rosenblatt).
// 'inputs' n'est PAS une liste de listes, mais un seul tableau 1D GÉANT contenant 
// toutes les images mises bout à bout. Cela optimise considérablement la vitesse de lecture en RAM.
std::vector<double> LinearModel::train(const std::vector<double>& inputs, const std::vector<double>& labels, double learning_rate, int epochs) {
    int input_size = weights.size();
    int num_samples = labels.size();

    std::vector<double> loss_history;

    // Répète l'apprentissage un certain nombre de fois (epochs) sur tout le dataset.
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int errors = 0; // Compteur d'erreurs pour l'epoch courante

        for (int i = 0; i < num_samples; ++i) {
            // 1. FORWARD PASS (Prédiction de l'image courante)
            // Le calcul est fait 'inline' plutôt que d'appeler predict() pour éviter
            // d'extraire/copier un sous-tableau de l'image courante, ce qui ferait chuter les perfs.
            double sum = bias;
            for (int j = 0; j < input_size; ++j) {
                // 'i * input_size + j' permet de retrouver le pixel 'j' de l'image 'i' dans le grand tableau 1D
                sum += weights[j] * inputs[i * input_size + j];
            }
            double prediction = (sum >= 0.0) ? 1.0 : -1.0;

            // 2. CALCUL DE L'ERREUR
            double error = labels[i] - prediction;

            // 3. BACKWARD PASS (Mise à jour des poids)
            // Si l'erreur n'est pas nulle (le modèle s'est trompé), on corrige les poids en les tirant 
            // vers la bonne direction, proportionnellement au taux d'apprentissage (learning_rate).
            if (error != 0.0) {
                errors++; // On comptabilise l'erreur
                for (int j = 0; j < input_size; ++j) {
                    weights[j] += learning_rate * error * inputs[i * input_size + j];
                }
                bias += learning_rate * error;
            }
        }

        // Enregistre le ratio d'erreurs de l'epoch (Loss) : 0.0 = parfait, 1.0 = tout faux
        loss_history.push_back(static_cast<double>(errors) / num_samples);
    }
    return loss_history;
}

// Entraînement depuis des chemins d'images : Charge, redimensionne et prépare la donnée avant d'entraîner.
std::vector<double> LinearModel::train_from_images(const std::vector<std::string>& image_paths, const std::vector<double>& labels, 
                                                   int target_w, int target_h, double learning_rate, int epochs) {
    std::vector<double> flattened_inputs;
    std::vector<double> valid_labels;

    for (size_t i = 0; i < image_paths.size(); ++i) {
        try {
            // On utilise c_str() au cas où la fonction attendrait un const char* standard
            std::vector<double> img_data = load_and_resize_image(image_paths[i].c_str(), target_w, target_h);
            
            // Ajout des pixels de l'image courante à la fin de l'immense tableau 1D
            flattened_inputs.insert(flattened_inputs.end(), img_data.begin(), img_data.end());
            
            // On conserve le label uniquement si l'image a été chargée avec succès
            valid_labels.push_back(labels[i]);
        } catch (const std::exception& e) {
            std::cerr << "  [C++] Erreur ignorée pour l'image " << image_paths[i] << " : " << e.what() << "\n";
        }
    }

    if (valid_labels.empty()) {
        throw std::runtime_error("Aucune image valide n'a pu être chargée pour l'entraînement.");
    }

    return train(flattened_inputs, valid_labels, learning_rate, epochs);
}

// Sauvegarde l'état du modèle dans un fichier texte.
// Format simple : la première ligne stocke le biais, et chaque ligne suivante stocke un poids.
void LinearModel::save(const char* filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << bias << "\n";
        for (double w : weights) {
            file << w << "\n";
        }
        file.close();
    } else {
        throw std::runtime_error(std::string("Erreur : Impossible de sauvegarder le fichier ") + filename);
    }
}

// Charge un modèle précédemment sauvegardé.
// Restaure le biais, nettoie les anciens poids, puis lit les nouveaux depuis le fichier.
void LinearModel::load(const char* filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        file >> bias;
        weights.clear();
        double w;
        while (file >> w) {
            weights.push_back(w);
        }
        file.close();
    } else {
        throw std::runtime_error(std::string("Erreur : Impossible de charger le fichier ") + filename);
    }
}

// Retourne le score mathématique brut de la prédiction (avant l'activation 1.0 ou -1.0).
// Plus la somme est grande (en positif ou négatif), plus le modèle est "confiant".
// Indispensable pour la stratégie One-Vs-Rest en Python pour classer 3 genres de jeux vidéo (Multi-classes).
double LinearModel::predict_raw(const std::vector<double>& inputs) const {
    double sum = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * inputs[i];
    }
    return sum;
}
