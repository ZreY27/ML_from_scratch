#pragma once
#include <vector>
#include <string>

class RBF {
private:
    int input_size;    // Dimension des entrées (ex: 32*32*3 pour une image)
    int num_centers;   // Nombre de neurones RBF dans la couche cachée (= nombre de centres)
    int output_size;   // Nombre de sorties (1 pour binaire, N pour multi-classe)
    double sigma;      // Largeur des gaussiennes (partagée entre tous les centres)
    bool is_classification;

    std::vector<std::vector<double>> centers;  // [num_centers][input_size] : positions des centres (k-means)
    std::vector<std::vector<double>> weights;  // [output_size][num_centers + 1] : poids de sortie (+1 pour le biais)

    // K-Means : place les centres en regroupant les données d'entrée en paquets.
    // Retourne, pour chaque sample, l'index du centre auquel il a été assigné.
    std::vector<int> kmeans(const std::vector<double>& flat_inputs, int num_samples, int max_iter = 300);

    // Calcule les activations de la couche cachée pour un seul sample.
    // phi[k] = exp(-||x - c_k||^2 / (2 * sigma^2)) : proche de 1 si x est près du centre k, proche de 0 sinon.
    std::vector<double> compute_phi(const std::vector<double>& x) const;

    // Calcule les poids de sortie par moindres carrés.
    // On résout le système (Phi^T * Phi) * W = Phi^T * Y par élimination de Gauss.
    // Phi : [num_samples][num_centers+1], Y : [num_samples][output_size]
    void fit_weights_least_squares(const std::vector<std::vector<double>>& Phi,
                                   const std::vector<std::vector<double>>& Y);

    // Si sigma n'est pas fourni, on en choisit un automatiquement à partir des centres :
    // sigma = d_max / sqrt(2 * num_centers), où d_max est la plus grande distance entre deux centres.
    double estimate_sigma() const;

public:
    // input_size   : Taille du vecteur d'entrée
    // num_centers  : Nombre de centres RBF (hyperparamètre clé)
    // output_size  : Nombre de sorties (1 = binaire, N = multi-classe one-hot)
    // sigma        : Largeur des gaussiennes (0.0 = estimation automatique)
    RBF(int input_size, int num_centers, int output_size = 1,
        double sigma = 0.0, bool is_classification = true);

    // Entraînement complet en deux phases :
    //   1. K-Means sur les inputs → centres
    //   2. Moindres carrés sur la couche de sortie → poids
    // inputs  : vecteur aplati [num_samples * input_size]
    // labels  : vecteur aplati [num_samples * output_size]
    // Retourne l'historique de loss (MSE ou taux d'erreur) sur quelques checkpoints
    std::vector<double> train(const std::vector<double>& inputs,
                              const std::vector<double>& labels,
                              int num_samples);

    // Entraînement direct depuis des chemins d'images
    std::vector<double> train_from_images(const std::vector<std::string>& image_paths,
                                          const std::vector<double>& labels,
                                          int target_w, int target_h);

    // Prédiction pour un seul sample (retourne output_size valeurs)
    // Classification : signe des sorties → +1 / -1 (binaire) ou argmax (multi-classe)
    // Régression     : valeurs brutes
    std::vector<double> predict(const std::vector<double>& inputs) const;

    // Retourne les activations brutes de la couche de sortie (avant signe/argmax)
    std::vector<double> predict_raw(const std::vector<double>& inputs) const;

    void save(const char* filename) const;
    void load(const char* filename);
};