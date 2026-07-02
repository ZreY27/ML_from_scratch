#pragma once
#include <vector>
#include <string>

// ============================================================================
// LinearModel — Modèle linéaire du cours :
//   - Classification : Perceptron, règle de Rosenblatt (slide 65 du cours
//     "Apprendre : Modèle Linéaire et PMC") avec sorties -1 / +1.
//   - Régression     : minimisation de l'erreur quadratique par descente de
//     gradient (le cours présente aussi la pseudo-inverse, slide 66).
// Auteur : Maxime Clément — partie individuelle : Modèle Linéaire, SVM,
//          cas de tests, scripts Python d'interfaçage.
// ============================================================================
class LinearModel {
private:
    std::vector<double> weights;
    double bias;

public:
    bool is_classification;

public:
    // input_size : Nombre d'entrées (ex: 32*32*3 pour une image)
    LinearModel(int input_size, bool is_classification = true);

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
