#pragma once
#include <vector>

class LinearModel {
private:
    std::vector<double> weights;
    double bias;

public:
    std::vector<double> loss_history;
    // input_size : Nombre d'entrées (ex: 32*32*3 pour une image)
    LinearModel(int input_size);

    // Forward pass
    double predict(const std::vector<double>& inputs) const;
    
    // Retourne la somme brute (score de confiance) pour le multi-classe (One-vs-Rest)
    double predict_raw(const std::vector<double>& inputs) const;
    // Backward pass
    // Retourne l'historique des erreurs (loss) par epoch
    void train(const std::vector<std::vector<double>> &X, const std::vector<double> &Y, double learning_rate, int epochs);
    
    void save(const char* filename);
    void load(const char* filename);
};
