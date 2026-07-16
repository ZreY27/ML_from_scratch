#include "ImageLoader.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

// Indique à STB de générer le code source de la bibliothèque ici
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {

// Valeur (0-255) du canal c du pixel source (x, y)
inline double px(const unsigned char* data, int w, int x, int y, int c) {
    return static_cast<double>(data[(y * w + x) * 3 + c]);
}

// Resize "area" (moyenne par zone, équivalent INTER_AREA d'OpenCV en réduction) :
// chaque pixel cible = moyenne de TOUS les pixels sources que sa zone recouvre,
// pondérée par le recouvrement fractionnaire aux bords. Aucun pixel source n'est
// ignoré -> conserve le plus d'information en forte réduction (contrairement à
// nearest qui ne lit qu'1 pixel source, ou bilinear qui n'en mélange que 4).
// En agrandissement (zone < 1 pixel source), se comporte comme du nearest.
void resize_area(const unsigned char* src, int w, int h,
                 std::vector<double>& out, int tw, int th) {
    const double x_ratio = static_cast<double>(w) / tw;
    const double y_ratio = static_cast<double>(h) / th;

    for (int i = 0; i < th; ++i) {
        const double y0 = i * y_ratio;
        const double y1 = (i + 1) * y_ratio;
        const int sy_first = static_cast<int>(y0);
        const int sy_last  = std::min(static_cast<int>(std::ceil(y1)), h) - 1;
        for (int j = 0; j < tw; ++j) {
            const double x0 = j * x_ratio;
            const double x1 = (j + 1) * x_ratio;
            const int sx_first = static_cast<int>(x0);
            const int sx_last  = std::min(static_cast<int>(std::ceil(x1)), w) - 1;

            double acc[3] = {0.0, 0.0, 0.0};
            for (int sy = sy_first; sy <= sy_last; ++sy) {
                const double wy = std::min(sy + 1.0, y1) - std::max(static_cast<double>(sy), y0);
                if (wy <= 0.0) continue;
                for (int sx = sx_first; sx <= sx_last; ++sx) {
                    const double wx = std::min(sx + 1.0, x1) - std::max(static_cast<double>(sx), x0);
                    if (wx <= 0.0) continue;
                    const double poids = wx * wy;
                    for (int c = 0; c < 3; ++c)
                        acc[c] += px(src, w, sx, sy, c) * poids;
                }
            }
            const double zone = (x1 - x0) * (y1 - y0);
            const int target_idx = (i * tw + j) * 3;
            for (int c = 0; c < 3; ++c)
                out[target_idx + c] = acc[c] / zone;
        }
    }
}

} // namespace

std::vector<double> load_and_resize_image(const std::string& filepath, int target_w, int target_h) {
    int w, h, channels;

    // Le "3" force la lecture en RGB, ignorant la transparence Alpha
    unsigned char* img_data = stbi_load(filepath.c_str(), &w, &h, &channels, 3);
    if (!img_data) {
        throw std::runtime_error("Erreur de chargement de l'image : " + filepath);
    }

    // Vecteur de sortie aplati (Largeur * Hauteur * 3), valeurs 0-255 après resize
    std::vector<double> result(target_w * target_h * 3);
    resize_area(img_data, w, h, result, target_w, target_h);

    stbi_image_free(img_data);

    // Normalisation des pixels (0-1)
    for (double& v : result)
        v /= 255.0;

    return result;
}
