#pragma once
#include <string>
#include <filesystem>

// Résout le chemin de sauvegarde/chargement d'un modèle, de façon robuste et portable.
//
//   - Nom de fichier simple ("mlp.txt")            -> préfixé "models/" : "models/mlp.txt"
//   - Chemin avec dossier ("models/mlp.txt",
//     "sous/dossier/x.txt", "models\\x.txt")        -> respecté tel quel
//   - Chemin absolu (Windows "C:\\..." ou "C:/...",
//     Unix "/...")                                  -> respecté tel quel
//
// Remplace l'ancienne logique `path.find("models/") != 0` qui cassait dès qu'on passait
// un chemin absolu ou avec des backslashes (produisait des chemins du type "models/C:\\Users\\...").
//
// Si create_dir == true (côté save), crée le dossier parent du chemin résolu si besoin.
inline std::string resolve_model_path(const std::string& filename, bool create_dir = false) {
    std::filesystem::path p(filename);

    // On ne préfixe "models/" que si l'utilisateur a donné un simple nom de fichier
    // (ni chemin absolu, ni composant de dossier).
    std::filesystem::path resolved =
        (p.is_absolute() || p.has_parent_path()) ? p : (std::filesystem::path("models") / p);

    if (create_dir) {
        std::filesystem::path dir = resolved.parent_path();
        if (!dir.empty())
            std::filesystem::create_directories(dir);
    }
    return resolved.string();
}
