#include "LinearModel.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

LinearModel::LinearModel(int input_size) : bias(0.0) { // on initialise le bias à 0
    weights.resize(input_size);//équivalent de : weights = malloc(input_size * sizeof(double)) avec une gestion dynamique de la mémoire
    for (int i = 0; i < input_size; i++) {
        weights[i] = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * 0.02 - 0.01; // initialisation aléatoire des poids entre -0.01 et +0.01
    }

}

double LinearModel::predict(const std::vector<double>& x) const {
    double sum = bias; // on commence la somme par le biais
    for (int i = 0; i < static_cast<int>(weights.size()); i++) { // on caste size retourne un unsigned
        sum += weights[i] * x[i]; // produit scalaire W.X
    }
    return sum >= 0.0 ? 1.0 : -1.0; // condition ternaire, si sum sup ou égale à 0, retourner 1 sinon -1
}

void LinearModel::train(const std::vector<std::vector<double>> &X,
                        const std::vector<double> &Y,
                        double learning_rate, int epochs) {
        for (int e = 0; e < epochs; e++) {
            int errors = 0;// compteur d'erreur
            for (int i = 0; i< static_cast<int>(X.size()); i++) { // boucle sur chaque exemple du dataset
                double pred =  predict(X[i]); //prédit la classe de l'exemple i
                double error = Y[i] - pred; //calcule l'erreur : si pred == y[i] → error = 0.0 → on ne fait rien si pred = -1 et y[i] = +1 → error = +2.0 → on corrige si pred = +1 et y[i] = -1 → error = -2.0 → on corrige
                if (error != 0.0) { //on ne met à jour les poids que si la prédiction est fausse, c'est la règle de Rosenblatt
                    errors ++;
                    for (int j = 0; j< static_cast<int>(weights.size()); j++) {
                        weights[j] += learning_rate * error * X[i][j];// si erreur > 0 : on augmente les poids des features positifs, si erreur < 0 : on les diminue
                    }
                    bias += learning_rate * error;//même règle pour le biais — il n'a pas de feature associé, donc on utilise juste lr * erreur (feature implicite = 1)
                }
            }
            loss_history.push_back(static_cast<double>(errors) / X.size()); // on stocke le taux d'erreur de cette epoch : nb_erreurs / nb_exemples, (double) = cast pour avoir une division décimale et pas entière, push_back() = ajoute à la fin du vector, comme append() en Python

        }
}

void LinearModel::save(const char* filename) {
    // TODO : sauvegarder weights et bias dans un fichier texte
}