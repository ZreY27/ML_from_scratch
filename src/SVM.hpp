#pragma once
#include <vector>
#include <string>

// ============================================================================
// SVM — Machine à Vecteurs de Support linéaire (classification), version
// "primal" : minimisation de la Hinge Loss + régularisation L2 par descente
// de (sous-)gradient. Maximiser la marge <=> minimiser ||W||^2 tout en
// pénalisant les exemples mal classés ou dans la marge (cf. slides 119-121).
//   NB : le cours présente la résolution duale (programmation quadratique,
//   alphas, vecteurs supports). Nous résolvons ici le problème PRIMAL,
//   équivalent pour le cas linéaire et plus simple à implémenter sans
//   solveur QP ; les non-linéarités sont gérées par transformation explicite
//   des entrées (pas de kernel trick, qui exige la formulation duale).
// Auteur : Maxime Clément — partie individuelle : Modèle Linéaire, SVM,
//          cas de tests, scripts Python d'interfaçage.
// ============================================================================
class SVM {
private:
    std::vector<double> weights;
    double bias;
    // lambda_reg : force de la régularisation L2 (pénalité sur ||W||^2).
    // Grand lambda -> priorité à une marge large (poids petits) ;
    // petit lambda -> priorité à bien classer les exemples.
    // NB : le "C" du soft-margin classique joue le rôle INVERSE (C ~ 1/lambda) :
    // il pénalise les violations de marge, pas les poids. On a nommé ce
    // paramètre lambda_reg pour éviter toute confusion avec le C du cours.
    double lambda_reg;

public:
    std::vector<double> loss_history;

    // input_size : nombre de features d'entrée
    // lambda_reg : régularisation L2 (défaut 0.001 : faible, proche d'un perceptron à marge)
    SVM(int input_size, double lambda_reg = 0.001);

    // Classe prédite : -1 ou +1 (signe de W.X + b)
    double predict(const std::vector<double>& x) const;

    // Retourne le score brut W.X + b (pour confiance / multi-classe One-vs-Rest)
    double predict_raw(const std::vector<double>& x) const;

    // Entraîne le SVM par descente de sous-gradient sur la Hinge Loss.
    // La loss moyenne par epoch est stockée dans loss_history.
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& Y,
               double learning_rate, int epochs);

    void save(const char* filename);
    void load(const char* filename);
};
