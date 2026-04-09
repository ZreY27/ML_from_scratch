#pragma once
#include <vector>

class LinearModel {
private:
    std::vector<double> weights;
    double bias;

public:
    // Mode du modèle
    enum Mode { CLASSIFICATION, REGRESSION };
    // CLASSIFICATION = retourne -1 ou +1 (Rosenblatt)
    // REGRESSION     = retourne une valeur continue (descente de gradient MSE)

    std::vector<double> loss_history;

    LinearModel(int input_size, Mode mode = CLASSIFICATION);
    // mode = CLASSIFICATION par défaut → tes tests actuels marchent sans changer

    double predict(const std::vector<double>& inputs) const;
    // en CLASSIFICATION → retourne -1.0 ou +1.0
    // en REGRESSION     → retourne la valeur brute W·X + b

    double predict_raw(const std::vector<double>& inputs) const;
    // retourne toujours la valeur brute, peu importe le mode

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<double>& Y,
               double learning_rate, int epochs);

    void save(const char* filename);
    void load(const char* filename);

private:
    Mode mode;
    // privé car l'extérieur n'a pas besoin de le modifier après construction
};