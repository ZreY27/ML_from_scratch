#include "RBF.hpp"
#include "ImageLoader.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <algorithm>

// Constructeur : on garde les hyperparamètres et on prépare les tableaux.
// Les centres et les poids sont mis à zéro ici, ils seront remplis pendant train().
RBF::RBF(int input_size, int num_centers, int output_size, double sigma, bool is_classification)
    : input_size(input_size),
      num_centers(num_centers),
      output_size(output_size),
      sigma(sigma),
      is_classification(is_classification)
{
    // Les centres et poids sont alloués au moment du train()
    // (leurs dimensions dépendent des données, pas seulement de input_size)
    centers.resize(num_centers, std::vector<double>(input_size, 0.0));

    // Poids [output_size][num_centers + 1] : le +1 sert pour le biais
    weights.resize(output_size, std::vector<double>(num_centers + 1, 0.0));
}

// Phase 1 de l'entraînement : K-Means.
// Le but est de répartir les centres au milieu des paquets de données.
std::vector<int> RBF::kmeans(const std::vector<double>& flat_inputs, int num_samples, int max_iter) {
    std::vector<int> assignments(num_samples, 0);

    // --- Initialisation : on choisit num_centers samples aléatoires comme centres initiaux ---
    std::vector<int> chosen_indices;
    for (int k = 0; k < num_centers; k++) {
        int idx;
        bool already_chosen;
        do {
            idx = rand() % num_samples;
            already_chosen = false;
            for (int c : chosen_indices)
                if (c == idx) { already_chosen = true; break; }
        } while (already_chosen);

        chosen_indices.push_back(idx);
        for (int j = 0; j < input_size; j++)
            centers[k][j] = flat_inputs[idx * input_size + j];
    }

    // --- Itérations K-Means ---
    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;

        // Étape E : assigner chaque sample au centre le plus proche
        for (int i = 0; i < num_samples; i++) {
            double best_dist = std::numeric_limits<double>::max();
            int best_k = 0;

            for (int k = 0; k < num_centers; k++) {
                double dist = 0.0;
                for (int j = 0; j < input_size; j++) {
                    double diff = flat_inputs[i * input_size + j] - centers[k][j];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            if (assignments[i] != best_k) {
                assignments[i] = best_k;
                changed = true;
            }
        }

        if (!changed) {
            std::cout << "\r  [K-Means] Convergence à l'itération " << iter + 1 << "          \n";
            break;
        }

        // Étape M : recalculer les centres comme moyenne des samples assignés
        std::vector<std::vector<double>> new_centers(num_centers, std::vector<double>(input_size, 0.0));
        std::vector<int> counts(num_centers, 0);

        for (int i = 0; i < num_samples; i++) {
            int k = assignments[i];
            counts[k]++;
            for (int j = 0; j < input_size; j++)
                new_centers[k][j] += flat_inputs[i * input_size + j];
        }

        for (int k = 0; k < num_centers; k++) {
            if (counts[k] > 0) {
                for (int j = 0; j < input_size; j++)
                    centers[k][j] = new_centers[k][j] / counts[k];
            }
            // Si un centre est vide (aucun sample assigné), on le réinitialise
            // avec un sample aléatoire pour éviter les centres morts
            else {
                int rand_idx = rand() % num_samples;
                for (int j = 0; j < input_size; j++)
                    centers[k][j] = flat_inputs[rand_idx * input_size + j];
            }
        }

        // Barre de progression K-Means
        int progress = (int)((float)(iter + 1) / max_iter * 100.0);
        std::cout << "\r  [K-Means] [";
        for (int p = 0; p < 50; ++p) {
            if (p < progress / 2) std::cout << "=";
            else if (p == progress / 2) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << progress << "%" << std::flush;
    }
    std::cout << std::endl;
    return assignments;
}

// Estimation automatique de sigma
// Choisit automatiquement une valeur de sigma quand l'utilisateur n'en donne pas.
// On prend la plus grande distance entre deux centres et on la met à l'échelle.
double RBF::estimate_sigma() const {
    // sigma = d_max / sqrt(2 * num_centers), avec d_max = plus grande distance entre deux centres
    double d_max = 0.0;
    for (int i = 0; i < num_centers; i++) {
        for (int j = i + 1; j < num_centers; j++) {
            double dist = 0.0;
            for (int k = 0; k < input_size; k++) {
                double diff = centers[i][k] - centers[j][k];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            if (dist > d_max) d_max = dist;
        }
    }
    // Garde-fou : si tous les centres sont confondus (dataset trop petit)
    if (d_max < 1e-10) d_max = 1.0;
    return d_max / std::sqrt(2.0 * num_centers);
}

// Calcule les activations de la couche cachée (les "phi") pour un sample.
std::vector<double> RBF::compute_phi(const std::vector<double>& x) const {
    // phi[k] = exp( -||x - c_k||^2 / (2 * sigma^2) )
    // On ajoute un biais en position 0 : phi[0] = 1.0
    std::vector<double> phi(num_centers + 1);
    phi[0] = 1.0; // biais

    double two_sigma_sq = 2.0 * sigma * sigma;

    for (int k = 0; k < num_centers; k++) {
        double dist_sq = 0.0;
        for (int j = 0; j < input_size; j++) {
            double diff = x[j] - centers[k][j];
            dist_sq += diff * diff;
        }
        phi[k + 1] = std::exp(-dist_sq / two_sigma_sq);
    }
    return phi;
}

// Phase 2 : on calcule les poids de sortie par moindres carrés.
// On résout (Phi^T * Phi) * W = Phi^T * Y au lieu de faire une descente de gradient.
void RBF::fit_weights_least_squares(const std::vector<std::vector<double>>& Phi,
                                    const std::vector<std::vector<double>>& Y) {
    int n = Phi.size();          // num_samples
    int m = Phi[0].size();       // num_centers + 1
    int q = Y[0].size();         // output_size

    // --- Calcul de A = Phi^T * Phi  [m x m] ---
    std::vector<std::vector<double>> A(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            for (int s = 0; s < n; s++)
                A[i][j] += Phi[s][i] * Phi[s][j];

    // On ajoute un petit lambda sur la diagonale pour éviter que la matrice
    // ne soit pas inversible (régularisation). Valeur faible pour ne pas trop fausser le résultat.
    double lambda = 1e-4;
    for (int i = 0; i < m; i++)
        A[i][i] += lambda;

    // --- Calcul de B = Phi^T * Y  [m x q] ---
    std::vector<std::vector<double>> B(m, std::vector<double>(q, 0.0));
    for (int i = 0; i < m; i++)
        for (int k = 0; k < q; k++)
            for (int s = 0; s < n; s++)
                B[i][k] += Phi[s][i] * Y[s][k];

    // --- Résolution de A * W = B par élimination de Gauss avec pivot partiel ---
    // On travaille sur une matrice augmentée [A | B] de taille [m x (m + q)]
    std::vector<std::vector<double>> aug(m, std::vector<double>(m + q, 0.0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++)
            aug[i][j] = A[i][j];
        for (int k = 0; k < q; k++)
            aug[i][m + k] = B[i][k];
    }

    // Pivot partiel
    for (int col = 0; col < m; col++) {
        // Trouver la ligne avec le pivot maximal
        int max_row = col;
        double max_val = std::abs(aug[col][col]);
        for (int row = col + 1; row < m; row++) {
            if (std::abs(aug[row][col]) > max_val) {
                max_val = std::abs(aug[row][col]);
                max_row = row;
            }
        }
        std::swap(aug[col], aug[max_row]);

        // Vérification de singularité
        if (std::abs(aug[col][col]) < 1e-12) {
            std::cerr << "  [RBF] Avertissement : matrice quasi-singulière à la colonne " << col << "\n";
            continue;
        }

        // Élimination
        double pivot = aug[col][col];
        for (int j = col; j < m + q; j++)
            aug[col][j] /= pivot;

        for (int row = 0; row < m; row++) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (int j = col; j < m + q; j++)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    // Extraction des poids : weights[k][j] = W[j][k]
    // Notre convention : weights[output][center+1]
    for (int k = 0; k < q; k++)
        for (int j = 0; j < m; j++)
            weights[k][j] = aug[j][m + k];
}

// Entraînement complet : d'abord K-Means pour les centres, puis moindres carrés pour les poids.
// inputs et labels sont aplatis en 1D (comme pour le LinearModel) pour aller plus vite.
std::vector<double> RBF::train(const std::vector<double>& inputs,
                               const std::vector<double>& labels,
                               int num_samples) {
    std::vector<double> loss_history;
    int labels_per_sample = output_size; // cohérence avec le format aplati des labels

    std::cout << "[RBF] Phase 1 : K-Means (" << num_centers << " centres)...\n";

    // --- Phase 1 : K-Means ---
    kmeans(inputs, num_samples);

    // --- Estimation / fixation de sigma ---
    if (sigma <= 0.0) {
        sigma = estimate_sigma();
        std::cout << "  [RBF] Sigma estimé automatiquement : " << sigma << "\n";
    }

    std::cout << "[RBF] Phase 2 : Calcul de la matrice Phi et moindres carrés...\n";

    // --- Construction de la matrice Phi [num_samples x (num_centers + 1)] ---
    std::vector<std::vector<double>> Phi(num_samples);
    for (int i = 0; i < num_samples; i++) {
        // Extraction du sample i depuis le vecteur aplati
        std::vector<double> x(inputs.begin() + i * input_size,
                               inputs.begin() + i * input_size + input_size);
        Phi[i] = compute_phi(x);

        // Progression
        if (i % (num_samples / 10 > 0 ? num_samples / 10 : 1) == 0) {
            int progress = (int)((float)i / num_samples * 100.0);
            std::cout << "\r  [Phi] [";
            for (int p = 0; p < 50; ++p) {
                if (p < progress / 2) std::cout << "=";
                else if (p == progress / 2) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << progress << "%" << std::flush;
        }
    }
    std::cout << "\r  [Phi] [==================================================] 100%\n";

    // --- Construction de Y [num_samples x output_size] ---
    std::vector<std::vector<double>> Y(num_samples, std::vector<double>(output_size));
    for (int i = 0; i < num_samples; i++)
        for (int k = 0; k < output_size; k++)
            Y[i][k] = labels[i * labels_per_sample + k];

    // --- Phase 2 : Moindres carrés ---
    fit_weights_least_squares(Phi, Y);

    // --- Calcul de la loss finale (MSE ou taux d'erreur) ---
    double total_loss = 0.0;
    int errors = 0;

    for (int i = 0; i < num_samples; i++) {
        std::vector<double> x(inputs.begin() + i * input_size,
                               inputs.begin() + i * input_size + input_size);
        std::vector<double> raw = predict_raw(x);

        for (int k = 0; k < output_size; k++) {
            double diff = Y[i][k] - raw[k];
            total_loss += diff * diff;
        }

        if (is_classification) {
            if (output_size == 1) {
                // Cas binaire : on regarde juste le signe de la sortie
                double pred = (raw[0] >= 0.0) ? 1.0 : -1.0;
                if (std::abs(pred - Y[i][0]) > 0.5) errors++;
            } else {
                // Cas multi-classe : on compare la classe gagnante (l'index du plus grand),
                // pas la valeur. C'est la même logique que dans predict().
                int pred_class = std::distance(raw.begin(), std::max_element(raw.begin(), raw.end()));
                int true_class = std::distance(Y[i].begin(), std::max_element(Y[i].begin(), Y[i].end()));
                if (pred_class != true_class) errors++;
            }
        }
    }

    double mse = total_loss / (num_samples * output_size);
    loss_history.push_back(mse);

    if (is_classification) {
        double error_rate = static_cast<double>(errors) / num_samples;
        std::cout << "[RBF] Entraînement terminé — MSE : " << mse
                  << " | Taux d'erreur : " << error_rate * 100.0 << "%\n";
        loss_history.push_back(error_rate);
    } else {
        std::cout << "[RBF] Entraînement terminé — MSE : " << mse << "\n";
    }

    return loss_history;
}

// Variante pratique : on donne directement les chemins des images.
// On les charge, on les met bout à bout, puis on appelle train().
std::vector<double> RBF::train_from_images(const std::vector<std::string>& image_paths,
                                            const std::vector<double>& labels,
                                            int target_w, int target_h) {
    int num_samples = static_cast<int>(image_paths.size());
    std::vector<double> flattened_inputs;
    flattened_inputs.reserve(num_samples * target_w * target_h * 3);

    int valid_samples = 0;
    std::vector<double> valid_labels;

    for (int i = 0; i < num_samples; i++) {
        try {
            std::vector<double> sample = load_and_resize_image(image_paths[i], target_w, target_h);
            flattened_inputs.insert(flattened_inputs.end(), sample.begin(), sample.end());

            // On copie output_size labels pour ce sample
            for (int k = 0; k < output_size; k++)
                valid_labels.push_back(labels[i * output_size + k]);

            valid_samples++;
        } catch (const std::exception& e) {
            std::cerr << "  [C++] Erreur ignorée pour l'image " << image_paths[i]
                      << " : " << e.what() << "\n";
        }
    }

    return train(flattened_inputs, valid_labels, valid_samples);
}

// Prédiction "brute" : on calcule les phi puis on fait weights * phi.
std::vector<double> RBF::predict_raw(const std::vector<double>& inputs) const {
    // Activation de la couche cachée
    std::vector<double> phi = compute_phi(inputs);

    // Couche de sortie : produit matriciel weights * phi
    std::vector<double> output(output_size, 0.0);
    for (int k = 0; k < output_size; k++)
        for (int j = 0; j < num_centers + 1; j++)
            output[k] += weights[k][j] * phi[j];

    return output;
}

std::vector<double> RBF::predict(const std::vector<double>& inputs) const {
    std::vector<double> raw = predict_raw(inputs);

    if (!is_classification)
        return raw; // Régression : retour direct

    if (output_size == 1) {
        // Binaire : signe → +1 ou -1
        return { raw[0] >= 0.0 ? 1.0 : -1.0 };
    } else {
        // Multi-classe : argmax → one-hot
        int best = static_cast<int>(std::distance(raw.begin(),
                                    std::max_element(raw.begin(), raw.end())));
        std::vector<double> one_hot(output_size, -1.0);
        one_hot[best] = 1.0;
        return one_hot;
    }
}

void RBF::save(const char* filename) const {
    std::string path = filename;
    // On préfixe models/ seulement si c'est un nom de fichier simple (pas un chemin absolu ou relatif)
    if (path.find('/') == std::string::npos)
        path = "models/" + path;

    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Erreur save RBF : impossible d'ouvrir " + path);

    // Métadonnées
    file << is_classification << "\n";
    file << input_size << " " << num_centers << " " << output_size << "\n";
    file << sigma << "\n";

    // Centres [num_centers x input_size]
    for (int k = 0; k < num_centers; k++) {
        for (int j = 0; j < input_size; j++)
            file << centers[k][j] << (j == input_size - 1 ? "" : " ");
        file << "\n";
    }

    // Poids [output_size x (num_centers + 1)]
    for (int k = 0; k < output_size; k++) {
        for (int j = 0; j < num_centers + 1; j++)
            file << weights[k][j] << (j == num_centers ? "" : " ");
        file << "\n";
    }

    file.close();
}

void RBF::load(const char* filename) {
    std::string path = filename;
    if (path.find("models/") != 0)
        path = "models/" + path;

    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Erreur load RBF : impossible d'ouvrir " + path);

    bool mode_val;
    file >> mode_val;
    is_classification = mode_val;

    file >> input_size >> num_centers >> output_size;
    file >> sigma;

    // Centres
    centers.assign(num_centers, std::vector<double>(input_size));
    for (int k = 0; k < num_centers; k++)
        for (int j = 0; j < input_size; j++)
            file >> centers[k][j];

    // Poids
    weights.assign(output_size, std::vector<double>(num_centers + 1));
    for (int k = 0; k < output_size; k++)
        for (int j = 0; j < num_centers + 1; j++)
            file >> weights[k][j];

    file.close();
    std::cout << "[RBF] Modèle chargé : " << path
              << " (" << num_centers << " centres, sigma=" << sigma << ")\n";
}