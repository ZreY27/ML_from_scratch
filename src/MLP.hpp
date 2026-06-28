#pragma once
#include <vector>
#include <string>

class MLP {
private:
    std::vector<int> d; // npl (Neurons Per Layer)
    int L; // Index de la dernière couche (len(npl) - 1)
    
    std::vector<std::vector<std::vector<double>>> W; // Poids W[layer][neurone_prev][neurone_actuel]
    std::vector<std::vector<double>> X; // Valeurs des neurones X[layer][neurone]
    std::vector<std::vector<double>> deltas; // Erreurs deltas[layer][neurone]
    bool is_classification; // Définit le mode (true = Classification, false = Régression)

    void propagate(const std::vector<double>& inputs);

public:
    MLP(const std::vector<int>& npl, bool is_classification = true);
    
    std::vector<double> predict(const std::vector<double>& inputs);
    
    // Les entrées et labels sont attendus aplatis (flattened 1D array) pour l'optimisation
    std::vector<double> train(const std::vector<double>& dataset_inputs, 
                              const std::vector<double>& dataset_expected_outputs,
                              int training_steps, 
                              double learning_rate, 
                              double decay = 0.0);
               
    // Entraînement direct depuis une liste de chemins d'images
    std::vector<double> train_from_images(const std::vector<std::string>& image_paths, 
                                          const std::vector<double>& expected_outputs,
                                          int target_w, 
                                          int target_h, 
                                          int training_steps, 
                                          double learning_rate, 
                                          double decay = 0.0);
               
    // Méthodes utilitaires de sérialisation
    void save(const char* filename);
    void load(const char* filename);
};