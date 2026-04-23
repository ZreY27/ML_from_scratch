#pragma once
#include <vector>

class MLP {
private:
    std::vector<int> d; // npl (Neurons Per Layer)
    int L; // Index de la dernière couche (len(npl) - 1)
    
    std::vector<std::vector<std::vector<double>>> W; // Poids W[layer][neurone_prev][neurone_actuel]
    std::vector<std::vector<double>> X; // Valeurs des neurones X[layer][neurone]
    std::vector<std::vector<double>> deltas; // Erreurs deltas[layer][neurone]

    void propagate(const std::vector<double>& inputs, bool is_classification);

public:
    MLP(const std::vector<int>& npl);
    
    std::vector<double> predict(const std::vector<double>& inputs, bool is_classification);
    
    // Les entrées et labels sont attendus aplatis (flattened 1D array) pour l'optimisation
    void train(const std::vector<double>& dataset_inputs, 
               const std::vector<double>& dataset_expected_outputs,
               int training_steps, 
               double learning_rate, 
               bool is_classification);
               
    // Méthodes utilitaires de sérialisation
    void save(const char* filename);
    void load(const char* filename);
};