#pragma once
#include <vector>
#include <string>

// Charge une image depuis le disque, la redimensionne (area averaging),
// l'aplatit (w*h*3) et la normalise (0-1).
//
// Resize "area" (équivalent INTER_AREA d'OpenCV) : chaque pixel cible = moyenne de
// TOUS les pixels sources qu'il recouvre. Choisi après comparaison avec nearest et
// bilinear : en forte réduction (ex. 640x360 -> 32x32, ~220 px source par px cible),
// c'est le seul qui n'ignore aucun pixel source -> conserve le plus d'information.
std::vector<double> load_and_resize_image(const std::string& filepath, int target_w, int target_h);
