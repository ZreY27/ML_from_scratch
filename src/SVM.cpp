// Auteur : Maxime Clément (voir SVM.h pour le détail de la partie individuelle)
#include "SVM.hpp"
#include "ModelPath.hpp"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>   // std::setprecision (sauvegarde sans perte)
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <random>     // std::mt19937 (mélange des exemples à chaque epoch)
#include <algorithm>  // std::shuffle

SVM::SVM(int input_size, double lambda_reg) : bias(0.0), lambda_reg(lambda_reg) {
    weights.resize(input_size);
    for (int i = 0; i < input_size; i++)
        weights[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 0.02 - 0.01;
}

double SVM::predict_raw(const std::vector<double> &x) const {
    // Garde-fou : une entrée de mauvaise taille provoquerait une lecture hors
    // limites ; pybind11 convertit l'exception en erreur Python lisible.
    if (x.size() != weights.size())
        throw std::invalid_argument("predict : l'entree a " + std::to_string(x.size()) +
                                    " valeurs mais le modele en attend " + std::to_string(weights.size()));

    double sum = bias;
    for (int i = 0; i < static_cast<int>(weights.size()); i++)
        sum += weights[i] * x[i];
    return sum;
}

double SVM::predict(const std::vector<double> &x) const {
    // Classe = signe de la distance (signée) à l'hyperplan
    return predict_raw(x) >= 0.0 ? 1.0 : -1.0;
}

void SVM::train(const std::vector<std::vector<double> > &X, const std::vector<double> &Y, double learning_rate, int epochs) {
    // Garde-fous : tailles cohérentes avant d'entraîner (sinon accès hors limites)
    if (X.empty())
        throw std::invalid_argument("train : dataset vide");
    if (X.size() != Y.size())
        throw std::invalid_argument("train : " + std::to_string(X.size()) + " exemples mais " +
                                    std::to_string(Y.size()) + " labels");
    for (size_t i = 0; i < X.size(); i++)
        if (X[i].size() != weights.size())
            throw std::invalid_argument("train : l'exemple " + std::to_string(i) + " a " +
                                        std::to_string(X[i].size()) + " valeurs, attendu " +
                                        std::to_string(weights.size()));

    // Un nouvel appel à train() repart d'un historique vide (sinon les courbes
    // de plusieurs entraînements successifs se mélangeraient).
    loss_history.clear();

    // Ordre de passage re-mélangé à chaque epoch (comme le LinearModel).
    // Indispensable : nos datasets arrivent triés PAR CLASSE ; sans mélange,
    // les blocs d'exemples consécutifs de la même classe font dériver les
    // poids cycliquement et l'entraînement peut s'effondrer.
    // Graine fixe (42) -> résultats reproductibles.
    std::vector<int> order(X.size());
    for (size_t i = 0; i < X.size(); i++) order[i] = static_cast<int>(i);
    std::mt19937 rng(42);

    for (int e = 0; e < epochs; e++) {
        std::shuffle(order.begin(), order.end(), rng);

        // Barre de progression (même style que LinearModel/MLP/RBF)
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

        double total_loss = 0.0;
        for (int i : order) {
            // Hinge Loss
            // margin = y * (W.X + b) : positif = bon cote de la frontiere,
            // et sa valeur = "a quelle distance" du bon cote on se trouve.
            // si margin >= 1 : bien classifie avec marge suffisante -> hinge = 0
            // si 0 < margin < 1 : bien classifie mais DANS le couloir -> petite penalite
            // si margin < 0 : mal classifie -> grosse penalite
            // Ex : Y=+1, score=+2.3 -> margin=+2.3 (parfait, hors couloir)
            //      Y=-1, score=+0.4 -> margin=-0.4 (mal classe : penalite 1.4)
            double score = predict_raw(X[i]);
            double margin = score * Y[i];
            double hinge = std::max(0.0, 1 - margin);
            total_loss += hinge;
            if (margin >= 1.0) {
                // Bien classifie avec marge : regularisation seulement
                // w = w - lr * 2 * lambda * w  (gradient de lambda*||w||^2)
                for (int j = 0; j < static_cast<int>(weights.size()); j++) {
                    weights[j] -= learning_rate * 2 * lambda_reg * weights[j];
                }
            } else {
                // Mal classifie ou dans la marge : regularisation + correction
                // w = w - lr * (2*lambda*w - y*x)   (sous-gradient de la hinge : -y*x)
                // b = b + lr * y
                for (int j = 0; j < static_cast<int>(weights.size()); j++) {
                    weights[j] -= learning_rate * (2 * lambda_reg * weights[j] - Y[i] * X[i][j]);
                }
                bias += learning_rate * Y[i];
            }
        }
        total_loss /= static_cast<double>(X.size());
        loss_history.push_back(total_loss);
    }
    std::cout << "\rTraining: [==================================================] 100% \n" << std::flush;
}

void SVM::save(const char *filename) {
    std::string path = resolve_model_path(filename, true);

    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Erreur save SVM");
    // 17 chiffres significatifs : un double est restitue a l\'identique au load()
    // (la precision par defaut de C++ est de 6 chiffres -> poids legerement degrades)
    file << std::setprecision(17);

    // En-tête "svm" : permet à load() de distinguer ce format des anciens
    // fichiers (qui commençaient par le mode 0/1 et contenaient un epsilon).
    file << "svm" << "\n";
    file << bias << "\n";
    file << lambda_reg << "\n";
    for (size_t i = 0; i < weights.size(); i++) {
        file << weights[i] << (i == weights.size() - 1 ? "" : " ");
    }
    file << "\n";
    file.close();
}

void SVM::load(const char *filename) {
    std::string path = resolve_model_path(filename);

    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Erreur load SVM : " + path);

    // Format actuel : "svm" \n bias \n lambda_reg \n poids...
    // Compat ascendante : les anciens fichiers commencent par le mode (0/1)
    // puis bias, lambda, epsilon (paramètre SVR abandonné, ignoré ici), poids.
    std::string premier_token;
    file >> premier_token;
    if (premier_token == "svm") {
        file >> bias >> lambda_reg;
    } else {
        double epsilon_ignore;
        file >> bias >> lambda_reg >> epsilon_ignore;
    }

    weights.clear();
    double w;
    while (file >> w) {
        weights.push_back(w);
    }
    file.close();
}
