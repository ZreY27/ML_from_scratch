#include "MLP.hpp"
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>

// Constructeur : Initialise l'architecture du réseau et alloue la mémoire.
// 'npl' (Neurons Per Layer) définit le nombre de neurones par couche (ex: {2, 3, 1}).
MLP::MLP(const std::vector<int>& npl) {
    d = npl;
    L = d.size() - 1;

    // Générateur de nombres aléatoires pour initialiser les poids entre -1.0 et 1.0
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Allocation des 3 structures principales du réseau (de la couche 0 à L)
    W.resize(L + 1);
    X.resize(L + 1);
    deltas.resize(L + 1);

    for (int l = 0; l <= L; ++l) {
        // +1 car l'index 0 est réservé au Biais
        X[l].resize(d[l] + 1, 0.0);
        deltas[l].resize(d[l] + 1, 0.0);
        
        for (int j = 0; j <= d[l]; ++j) {
            // Le Biais est l'astuce mathématique du réseau : le neurone 0 vaut TOUJOURS 1.0
            X[l][j] = (j == 0) ? 1.0 : 0.0; 
        }

        if (l == 0) continue;

        W[l].resize(d[l - 1] + 1);
        for (int i = 0; i <= d[l - 1]; ++i) {
            W[l][i].resize(d[l] + 1, 0.0);
            for (int j = 1; j <= d[l]; ++j) {
                W[l][i][j] = dis(gen); // Poids aléatoire de la connexion i (couche l-1) vers j (couche l)
            }
        }
    }
}

// Forward Pass (Propagation avant) : Calcule la prédiction du réseau pour une entrée donnée.
// Traverse les couches une par une, fait la somme pondérée, et applique la fonction d'activation.
void MLP::propagate(const std::vector<double>& inputs, bool is_classification) {
    // 1. Assigner les valeurs d'entrée à la première couche (couche 0)
    for (int j = 1; j <= d[0]; ++j) {
        X[0][j] = inputs[j - 1];
    }

    // 2. Propager le signal à travers les couches cachées et la couche de sortie
    for (int l = 1; l <= L; ++l) {
        for (int j = 1; j <= d[l]; ++j) {
            double total = 0.0;
            // Somme pondérée (inclut implicitement le biais car X[l-1][0] = 1.0)
            for (int i = 0; i <= d[l - 1]; ++i) {
                total += W[l][i][j] * X[l - 1][i];
            }
            // Fonction d'activation Tangente Hyperbolique (tanh)
            // Appliquée sur toutes les couches cachées, et sur la sortie si classification
            if (is_classification || l < L) {
                total = std::tanh(total);
            }
            X[l][j] = total;
        }
    }
}

// Prédiction publique : Appelle propagate() et retourne uniquement le résultat
// de la dernière couche sous forme de vecteur (utile pour le multi-classes).
std::vector<double> MLP::predict(const std::vector<double>& inputs, bool is_classification) {
    propagate(inputs, is_classification);
    std::vector<double> result(d[L]);
    for (int j = 1; j <= d[L]; ++j) {
        result[j - 1] = X[L][j];
    }
    return result;
}

// Entraînement du modèle (Algorithme de Rétropropagation du Gradient / Backpropagation)
// Utilise la Descente de Gradient Stochastique (Stochastic Gradient Descent - SGD).
// Prends des listes 1D aplaties pour des performances optimales (évite les copies mémoire).
void MLP::train(const std::vector<double>& dataset_inputs, const std::vector<double>& dataset_expected_outputs,
                int training_steps, double learning_rate, bool is_classification) {
                
    int num_samples = dataset_inputs.size() / d[0];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_k(0, num_samples - 1);

    for (int step = 0; step < training_steps; ++step) {
        // 1. Choix d'un échantillon aléatoire (SGD)
        int k = dist_k(gen);
        
        // Extraction dynamique de l'échantillon depuis les listes aplaties
        std::vector<double> inputs_k(d[0]);
        for(int j = 0; j < d[0]; ++j) {
            inputs_k[j] = dataset_inputs[k * d[0] + j];
        }
        
        std::vector<double> y_k(d[L]);
        for(int j = 0; j < d[L]; ++j) {
            y_k[j] = dataset_expected_outputs[k * d[L] + j];
        }

        // 2. FORWARD PASS : Le réseau tente de deviner
        propagate(inputs_k, is_classification);

        // 3. BACKWARD PASS (Étape 1/2) : Calcul de l'erreur (Deltas) sur la DERNIÈRE couche
        for (int j = 1; j <= d[L]; ++j) {
            deltas[L][j] = X[L][j] - y_k[j - 1]; // Erreur basique : (Prédiction - Attendu)
            if (is_classification) {
                // Multiplié par la dérivée de tanh (qui est 1 - tanh^2)
                deltas[L][j] *= (1.0 - X[L][j] * X[L][j]);
            }
        }

        // 4. BACKWARD PASS (Étape 2/2) : Rétropropagation de l'erreur dans les COUCHES CACHÉES
        for (int l = L; l >= 2; --l) {
            for (int i = 1; i <= d[l - 1]; ++i) {
                double total = 0.0;
                // On récupère la somme des erreurs de la couche suivante pondérées par les poids
                for (int j = 1; j <= d[l]; ++j) {
                    total += W[l][i][j] * deltas[l][j];
                }
                // Multiplié par la dérivée de l'activation (1 - tanh^2) du neurone courant
                total *= (1.0 - X[l - 1][i] * X[l - 1][i]);
                deltas[l - 1][i] = total;
            }
        }

        // 5. MISE À JOUR DES POIDS : Descente de gradient
        for (int l = 1; l <= L; ++l) {
            for (int i = 0; i <= d[l - 1]; ++i) {
                for (int j = 1; j <= d[l]; ++j) {
                    // Nouveau poids = Ancien poids - (LearningRate * SortieNeuronePrécédent * DeltaNeuroneActuel)
                    W[l][i][j] -= learning_rate * X[l - 1][i] * deltas[l][j];
                }
            }
        }
    }
}

// Sauvegarde l'architecture complète du modèle (npl) et ses poids dans un fichier texte.
// La ligne 1 contient l'architecture (ex: "2 3 1"). Les lignes suivantes contiennent les poids.
void MLP::save(const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Erreur save MLP");
    for (int size : d) file << size << " ";
    file << "\n";
    for (int l = 1; l <= L; ++l) {
        for (int i = 0; i <= d[l - 1]; ++i) {
            for (int j = 1; j <= d[l]; ++j) {
                file << W[l][i][j] << " ";
            }
            file << "\n";
        }
    }
    file.close();
}

// Charge un modèle depuis un fichier sauvegardé.
// Détruit l'architecture courante, lit l'architecture sauvegardée sur la ligne 1,
// réalloue la bonne quantité de mémoire, puis injecte les poids.
void MLP::load(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error(std::string("Erreur : Impossible de charger le fichier ") + filename);

    // 1. Lire la première ligne pour reconstruire l'architecture (tableau d)
    std::string line;
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        d.clear();
        int size;
        while (ss >> size) {
            d.push_back(size);
        }
    }
    
    L = d.size() - 1;

    // 2. Réallouer la mémoire pour l'architecture chargée
    W.clear(); X.clear(); deltas.clear();
    W.resize(L + 1); X.resize(L + 1); deltas.resize(L + 1);

    for (int l = 0; l <= L; ++l) {
        X[l].resize(d[l] + 1, 0.0);
        deltas[l].resize(d[l] + 1, 0.0);
        X[l][0] = 1.0; // Biais toujours à 1.0 à l'index 0

        if (l == 0) continue;

        W[l].resize(d[l - 1] + 1);
        for (int i = 0; i <= d[l - 1]; ++i) {
            W[l][i].resize(d[l] + 1, 0.0);
            // 3. Lire et restaurer chaque poids individuellement
            for (int j = 1; j <= d[l]; ++j) {
                file >> W[l][i][j];
            }
        }
    }
    file.close();
}