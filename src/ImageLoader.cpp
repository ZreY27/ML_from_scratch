#include "ImageLoader.hpp"
#include <stdexcept>
#include <cmath>

// Indique à STB de générer le code source de la bibliothèque ici
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::vector<double> load_and_resize_image(const std::string& filepath, int target_w, int target_h) {
    int w, h, channels;
    
    // Le "3" force la lecture en RGB, ignorant la transparence Alpha
    unsigned char* img_data = stbi_load(filepath.c_str(), &w, &h, &channels, 3);
    if (!img_data) {
        throw std::runtime_error("Erreur de chargement de l'image : " + filepath);
    }

    // Vecteur de sortie aplati (Largeur * Hauteur * 3)
    std::vector<double> result(target_w * target_h * 3);

    // Algorithme de redimensionnement "Nearest Neighbor" (Plus Proche Voisin)
    double x_ratio = w / (double)target_w;
    double y_ratio = h / (double)target_h;

    for (int i = 0; i < target_h; i++) {
        for (int j = 0; j < target_w; j++) {
            int px = std::floor(j * x_ratio);
            int py = std::floor(i * y_ratio);
            
            int original_idx = (py * w + px) * 3;
            int target_idx = (i * target_w + j) * 3;
            
            // Normalisation des pixels (divisé par 255.0)
            result[target_idx] = img_data[original_idx] / 255.0;         // Rouge
            result[target_idx + 1] = img_data[original_idx + 1] / 255.0; // Vert
            result[target_idx + 2] = img_data[original_idx + 2] / 255.0; // Bleu
        }
    }

    stbi_image_free(img_data);

    return result;
}