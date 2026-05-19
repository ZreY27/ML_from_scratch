#pragma once
#include <vector>
#include <string>

class SVM {
private:
    std::vector<double> weights;
    double bias;
    double C;

public:
    std::vector<double> loss_history;
    //C=1 équivaut à la bonne valeur de départ, input_size est le nombre de features d'entree
    SVM(int input_size, double C = 1.0 );
    // -1 ou 1
    double predict(const std::vector<double>& x) const;

    // Retourne le score brut W.X + b (pour confiance / multi-classe)
    double predict_raw(const std::vector<double>& x) const;

    // Entraine le SVM par descente de gradient sur la Hinge Loss
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& Y,
               double learning_rate, int epochs);

    void save(const char* filename);
    void load(const char* filename);
};
