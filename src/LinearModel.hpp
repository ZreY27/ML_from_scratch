#pragma once
#include <vector>
#include <string>

class LinearModel {
private:
    std::vector<double> weights;
    double bias;

public:
    enum Mode { CLASSIFICATION, REGRESSION };

private:
    Mode mode;

public:
    // input_size : Nombre d'entrées (ex: 32*32*3 pour une image)
    LinearModel(int input_size, Mode mode = CLASSIFICATION);

    // Forward pass
    double predict(const std::vector<double>& inputs) const;
    
    // Retourne la somme brute (score de confiance) pour le multi-classe (One-vs-Rest)
    double predict_raw(const std::vector<double>& inputs) const;

    // Backward pass
    // Retourne l'historique des erreurs (loss) par epoch
    std::vector<double> train(const std::vector<double>& inputs, const std::vector<double>& labels, double learning_rate, int epochs);
    
    // Entraînement direct depuis une liste de chemins d'images
    std::vector<double> train_from_images(const std::vector<std::string>& image_paths, const std::vector<double>& labels, 
                                          int target_w, int target_h, double learning_rate, int epochs);

    void save(const char* filename);
    void load(const char* filename);
};
