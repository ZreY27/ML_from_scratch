#pragma once  // Toujours mettre ça en première ligne pour éviter les inclusions multiples
#include <vector>

class LinearModel {
public:
    std::vector<double> weights; //correspond à un tableau à taille variable, celui du poids du modèle
    double bias;
    std::vector<double> loss_history; // taux d'erreur à chaque epoch
    // 1. Le Constructeur : Créer le modèle
    // input_size : Nombre d'entrées (ex: 32*32*3 pour une image)
    LinearModel(int input_size);

    // 2. Prédiction : Estime une valeur (Forward pass)
    // "const ... &" signifie : "Je lis le vecteur sans le copier" (Optimisation Vitesse)
    double predict(const std::vector<double>& x) const;

    // 3. Entraînement : Corrige les poids (Backward pass)
    void train(const std::vector<std::vector<double>> &X,
               const std::vector<double> &Y,
               double learning_rate, int epochs);
    // X datasetcomplet (matrice), Y labels correspondant (-1 ou 1), epochs = nombre de passes complètes sur le dataset
    
    // 4. Sauvegarde
    void save(const char* filename);
};