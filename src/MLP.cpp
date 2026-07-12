// ============================================================================
// MLP — Perceptron Multi-Couches entraîné par rétropropagation du gradient
// stochastique (cf. slides 87-95 du cours "Apprendre : Modèle Linéaire et PMC").
// Auteur : Antoine (partie individuelle : PMC / MLP)
// ============================================================================
#include "MLP.hpp"
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>   // std::setprecision (sauvegarde sans perte)
#include <stdexcept>
#include <sstream>
#include <string>
#include <iostream>
#include "ImageLoader.hpp"
#include "ModelPath.hpp"
#include <filesystem>

// Constructeur : Initialise l'architecture du réseau et alloue la mémoire.
// 'npl' (Neurons Per Layer) définit le nombre de neurones par couche (ex: {2, 3, 1}).
MLP::MLP(const std::vector<int>& npl, bool is_classification) : is_classification(is_classification) {
    d = npl;
    L = d.size() - 1;

    // Générateur de nombres aléatoires pour initialiser les poids entre -1.0 et 1.0.
    // Graine = 42 + compteur d'instances : chaque MLP construit reçoit une graine
    // DIFFÉRENTE (42, 43, 44...) mais la séquence est REPRODUCTIBLE d'une exécution
    // à l'autre. Indispensable pour le seeder/bagging : les N variants doivent partir
    // d'initialisations différentes (sinon ils seraient identiques et la moyenne des
    // sorties n'apporterait rien). Une graine fixe unique rendrait tous les variants
    // identiques -> variance nulle, bagging inutile.
    static unsigned int compteur_instances = 0;
    std::mt19937 gen(42 + compteur_instances++);
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
void MLP::propagate(const std::vector<double>& inputs) {
    // Garde-fou : une entrée de mauvaise taille provoquerait une lecture hors
    // limites ; pybind11 convertit l'exception en erreur Python lisible.
    if (static_cast<int>(inputs.size()) != d[0])
        throw std::invalid_argument("predict : l'entree a " + std::to_string(inputs.size()) +
                                    " valeurs mais le reseau en attend " + std::to_string(d[0]));

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
std::vector<double> MLP::predict(const std::vector<double>& inputs) {
    propagate(inputs);
    std::vector<double> result(d[L]);
    for (int j = 1; j <= d[L]; ++j) {
        result[j - 1] = X[L][j];
    }
    return result;
}

// Entraînement du modèle (Algorithme de Rétropropagation du Gradient / Backpropagation)
// Utilise la Descente de Gradient Stochastique (Stochastic Gradient Descent - SGD).
// Prends des listes 1D aplaties pour des performances optimales (évite les copies mémoire).
std::vector<double> MLP::train(const std::vector<double>& dataset_inputs, const std::vector<double>& dataset_expected_outputs,
                               int training_steps, double learning_rate, double decay) {
                
    int num_samples = static_cast<int>(dataset_inputs.size()) / d[0];

    // Garde-fous : tailles cohérentes avant d'entraîner (sinon accès hors limites)
    if (num_samples == 0)
        throw std::invalid_argument("train : dataset vide ou plus petit qu'un seul exemple");
    if (dataset_inputs.size() % d[0] != 0)
        throw std::invalid_argument("train : inputs contient " + std::to_string(dataset_inputs.size()) +
                                    " valeurs, non divisible par la taille d'entree " + std::to_string(d[0]));
    if (dataset_expected_outputs.size() != static_cast<size_t>(num_samples) * static_cast<size_t>(d[L]))
        throw std::invalid_argument("train : outputs contient " + std::to_string(dataset_expected_outputs.size()) +
                                    " valeurs, attendu " + std::to_string(num_samples) + " x " + std::to_string(d[L]));

    // Graine fixe (42) -> même séquence d'échantillons SGD à chaque exécution (reproductible)
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist_k(0, num_samples - 1);
    std::vector<double> loss_history;

    for (int step = 0; step < training_steps; ++step) {
        if (step % (training_steps / 100 > 0 ? training_steps / 100 : 1) == 0 || step == training_steps - 1) {
            int progress = (int)((float)step / training_steps * 100.0);
            std::cout << "\rTraining: [";
            for (int p = 0; p < 50; ++p) {
                if (p < progress / 2) std::cout << "=";
                else if (p == progress / 2) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << progress << "% " << std::flush;
        }
        
        // Application du Learning Rate Decay (décroissance du taux d'apprentissage)
        double current_learning_rate = learning_rate * (1.0 / (1.0 + decay * step));

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
        propagate(inputs_k);

        // 3. BACKWARD PASS (Étape 1/2) : Calcul de l'erreur (Deltas) sur la DERNIÈRE couche
        double step_loss = 0.0;
        for (int j = 1; j <= d[L]; ++j) {
            double err = X[L][j] - y_k[j - 1]; // Erreur basique : (Prédiction - Attendu)
            step_loss += err * err; // MSE
            deltas[L][j] = err; 
            if (is_classification) {
                // Multiplié par la dérivée de tanh (qui est 1 - tanh^2)
                deltas[L][j] *= (1.0 - X[L][j] * X[L][j]);
            }
        }
        loss_history.push_back(step_loss / d[L]);

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
                    // Nouveau poids = Ancien poids - (CurrentLearningRate * SortieNeuronePrécédent * DeltaNeuroneActuel)
                    W[l][i][j] -= current_learning_rate * X[l - 1][i] * deltas[l][j];
                }
            }
        }
    }
    std::cout << std::endl;
    return loss_history;
}

// Entraînement direct depuis des chemins d'images : Charge, redimensionne et prépare la donnée avant d'entraîner.
std::vector<double> MLP::train_from_images(const std::vector<std::string>& image_paths, const std::vector<double>& expected_outputs,
                                           int target_w, int target_h, int training_steps, double learning_rate, double decay) {
    std::vector<double> flattened_inputs;
    std::vector<double> valid_expected_outputs;
    int output_size = d[L];

    // Garde-fou : output_size labels par image, sinon l'accès expected_outputs[i * output_size + j] déborde.
    if (expected_outputs.size() != image_paths.size() * static_cast<size_t>(output_size))
        throw std::invalid_argument("train_from_images : " + std::to_string(image_paths.size()) +
                                    " images mais " + std::to_string(expected_outputs.size()) +
                                    " labels (attendu " + std::to_string(output_size) + " par image)");

    for (size_t i = 0; i < image_paths.size(); ++i) {
        try {
            std::vector<double> img_data = load_and_resize_image(image_paths[i].c_str(), target_w, target_h);
            
            // Ajout des pixels de l'image courante
            flattened_inputs.insert(flattened_inputs.end(), img_data.begin(), img_data.end());
            
            // Copie des sorties attendues (labels) correspondantes pour cette image
            for (int j = 0; j < output_size; ++j) {
                valid_expected_outputs.push_back(expected_outputs[i * output_size + j]);
            }
        } catch (const std::exception& e) {
            std::cerr << "  [C++] Erreur ignorée pour l'image " << image_paths[i] << " : " << e.what() << "\n";
        }
    }

    if (valid_expected_outputs.empty()) {
        throw std::runtime_error("Aucune image valide n'a pu être chargée pour l'entraînement.");
    }

    return train(flattened_inputs, valid_expected_outputs, training_steps, learning_rate, decay);
}

// Sauvegarde l'architecture complète du modèle (npl) et ses poids dans un fichier texte.
// La ligne 1 contient l'architecture (ex: "2 3 1"). Les lignes suivantes contiennent les poids.
void MLP::save(const char* filename) {
    std::string path = resolve_model_path(filename, true);

    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Erreur save MLP");
    // 17 chiffres significatifs : un double est restitue a l\'identique au load()
    // (la precision par defaut de C++ est de 6 chiffres -> poids legerement degrades)
    file << std::setprecision(17);
    // En-tête : persiste le mode (classification / régression) sur sa propre ligne,
    // AVANT l'architecture. Voir load() pour la compatibilité avec les anciens fichiers.
    file << "mode " << is_classification << "\n";
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
    std::string path = resolve_model_path(filename);

    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error(std::string("Erreur : Impossible de charger le fichier ") + path);

    // 1. Lire l'en-tête. Format actuel : une ligne "mode <0|1>" précède l'architecture.
    //    Compat ascendante : les anciens fichiers commencent directement par l'architecture
    //    (pas de "mode") → on conserve alors le is_classification courant (celui du constructeur).
    std::string line;
    std::getline(file, line);
    {
        std::stringstream head(line);
        std::string tok;
        head >> tok;
        if (tok == "mode") {
            int m = 1;
            head >> m;
            is_classification = (m != 0);
            std::getline(file, line); // la ligne suivante contient l'architecture (npl)
        }
    }

    // 2. Reconstruire l'architecture (tableau d) depuis la ligne npl
    std::stringstream ss(line);
    d.clear();
    int size;
    while (ss >> size) {
        d.push_back(size);
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