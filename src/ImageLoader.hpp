#pragma once
#include <vector>
#include <string>

// Charge une image depuis le disque, la redimensionne, l'aplatit et la normalise (0-1)
std::vector<double> load_and_resize_image(const std::string& filepath, int target_w, int target_h);