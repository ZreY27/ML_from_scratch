# Révision — MLP de classification (soutenance)

Matériel de révision pour la partie individuelle **MLP/PMC** (C++ from scratch + pybind11).
Tâche : classer des captures de jeux vidéo en 3 genres (Fighter, Platformer, Racing).

## Documents

| Document | Contenu |
|---|---|
| [`justification_hyperparametres.md`](justification_hyperparametres.md) | **Pourquoi ces hyperparamètres ?** Rapport structuré Problème → Analyse → Observation → Déduction → Conclusion, sur **tous les modèles générés** (15 runs). Tableau final : chaque hyperparamètre = valeur + statut (validé / raisonné / perfectible) + justification. |
| [`analyse_sweep_couche_cachee.md`](analyse_sweep_couche_cachee.md) | **Analyse détaillée du sweep d'architecture** : les 5 runs à 300k/3 clones (§1–4) + le sweep exploratoire 10 runs/4 axes (§5). Diagnostic chiffré de la saturation de `tanh`. |

## Les 3 messages clés
1. **h128 est le point optimal** (85,2 % test) — la performance n'est pas monotone avec la taille du réseau.
2. **lr = 0.01 et 300k steps sont des optima mesurés** (lr 0.1 diverge, 0.001 sous-apprend ; l'accuracy croît jusqu'à 300k sans sur-apprentissage).
3. **Les gros réseaux échouent par instabilité d'initialisation** (`U(−1,1)` non mise à l'échelle du fan-in → saturation `tanh` → collapse vers une classe à 33 %). Correctif identifié : **Xavier/Glorot**.

## Reproduire les données
```bash
tensorboard --logdir runs           # visualiser les 15 runs (mlp_* + exp_*)
python scripts/sweep_mlp_hidden.py  # sweep principal (5 runs, 300k, 3 clones)

# sweep exploratoire : train_one.py = 1 config ; paralléliser avec xargs (-P = nb de cœurs)
printf '%s\n' "h128 128 0.01 100000" "h32 32 0.01 100000" "lr0.1 128 0.1 100000" \
  | xargs -P4 -n4 sh -c 'python scripts/train_one.py --name "exp_$1" --arch "$2" --lr "$3" --steps "$4"' _
```
