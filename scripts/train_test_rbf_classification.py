import os
import time
import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Autorise Python à charger les DLLs du compilateur C++ (MSYS2)
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r"C:\msys64\ucrt64\bin")

# Dossier racine du projet (pour accéder aux datasets)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

import ML_ESGI

# 1. Paramètres du modèle
IMAGE_WIDTH  = 32
IMAGE_HEIGHT = 32
INPUT_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT * 3  # 3072 pixels

# Hyperparamètres RBF
NUM_CENTERS = 200   # Nombre de centres K-Means
SIGMA       = 0.0  # 0.0 = estimation automatique depuis les centres

# 2. Définition des catégories et de leurs labels (One-Hot Encoding)
# Même convention que le MLP : un neurone par catégorie, +1 pour la gagnante, -1 pour les autres
categories = ["Fighter", "Racing", "Platformer"]
category_labels = {
    "Fighter":    [ 1.0, -1.0, -1.0],
    "Racing":     [-1.0,  1.0, -1.0],
    "Platformer": [-1.0, -1.0,  1.0]
}

all_paths      = []
all_labels_flat = []

# 3. Parcours des dossiers pour lister toutes les images
for cat in categories:
    search_path_jpg = os.path.join(ROOT_DIR, "datasets", cat, "*.jpg")
    search_path_png = os.path.join(ROOT_DIR, "datasets", cat, "*.png")
    imgs = glob.glob(search_path_jpg) + glob.glob(search_path_png)

    for img_path in imgs:
        all_paths.append(img_path)
        all_labels_flat.extend(category_labels[cat])

if len(all_paths) == 0:
    print("Aucune image trouvée. Vérifiez les dossiers 'datasets/Fighter', etc.")
else:
    print(f"{len(all_paths)} images trouvées. Préparation de l'entraînement RBF...")

    # 4. Création du modèle RBF
    # input_size  : taille d'une image aplatie (3072)
    # num_centers : nombre de centres (= neurones cachés)
    # output_size : 3 sorties, une par catégorie
    # sigma       : 0.0 pour que le modèle le choisisse tout seul
    model = ML_ESGI.RBF(INPUT_SIZE, NUM_CENTERS, output_size=3, sigma=SIGMA, is_classification=True)

    print("Début de l'entraînement (K-Means puis moindres carrés)...")
    start_time = time.time()
    loss_history = model.train_from_images(
        image_paths=all_paths,
        labels=all_labels_flat,
        target_w=IMAGE_WIDTH,
        target_h=IMAGE_HEIGHT
    )

    elapsed = time.time() - start_time
    print(f"Temps d'entraînement : {elapsed:.2f} secondes")

    # 5. Envoi des résultats à TensorBoard
    writer = SummaryWriter("runs/rbf")
    mse = loss_history[0]
    writer.add_scalar("Résultats/MSE", mse, 0)
    if len(loss_history) > 1:
        writer.add_scalar("Résultats/Taux_erreur", loss_history[1] * 100.0, 0)
        writer.add_scalar("Résultats/Précision", 100.0 - loss_history[1] * 100.0, 0)
    writer.close()

    # 6. Sauvegarde du modèle entraîné
    save_path = os.path.join(ROOT_DIR, "models", "rbf_params.txt")
    model.save(save_path)
    print(f"\nEntraînement terminé ! Modèle sauvegardé dans '{save_path}'")

    # 6. Affichage du résultat
    # Le RBF n'a pas d'historique par epoch (il s'entraîne en une seule passe),
    # donc on affiche juste la MSE finale et le taux d'erreur sous forme de texte.
    mse = loss_history[0]
    if len(loss_history) > 1:
        error_rate = loss_history[1] * 100.0
        print(f"MSE finale    : {mse:.4f}")
        print(f"Taux d'erreur : {error_rate:.1f}%")
        print(f"Précision     : {100.0 - error_rate:.1f}%")

        precision = 100.0 - error_rate
        plt.pie(
            [precision, error_rate],
            labels=[f"Réussite ({precision:.1f}%)", f"Erreur ({error_rate:.1f}%)"],
            colors=["mediumseagreen", "tomato"],
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title("Résultats du RBF après entraînement")
        result_path = os.path.join(ROOT_DIR, "results", "rbf_resultat.png")
        plt.savefig(result_path)
        print(f"Graphique sauvegardé dans '{result_path}'")
        plt.show()
    else:
        print(f"MSE finale : {mse:.4f}")
        plt.bar(["MSE"], [mse], color=["steelblue"])
        plt.title("Résultat du RBF (régression)")
        plt.ylabel("MSE")
        plt.show()
