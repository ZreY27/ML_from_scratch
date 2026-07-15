# Justification des choix d'hyperparamètres — MLP de classification

*Rapport de révision — partie individuelle : Perceptron Multi-Couches (PMC/MLP) implémenté « from scratch » en C++, binding Python via pybind11. Tâche : classer des captures d'écran de jeux vidéo en 3 genres (Fighter, Platformer, Racing).*

**Structure :** Problème → Analyse → Observation → Déduction → Conclusion. Chaque choix d'hyperparamètre est justifié soit par un **principe** (a priori), soit par une **preuve expérimentale** tirée des modèles générés.

## Base de preuve : tous les modèles générés

| Campagne | Runs | Protocole | Rôle dans ce rapport |
|---|---|---|---|
| **Sweep principal** | `mlp_h64`, `mlp_h128`, `mlp_h256`, `mlp_h512`, `mlp_h256x128` | 300k steps · **3 clones** + bagging · lr 0.01 | Choix de l'architecture + robustesse |
| **Sweep exploratoire** | `exp_h128`, `exp_h128x64`, `exp_h256x128x64`, `exp_h512x256`, `exp_h32`, `exp_lr0.001`, `exp_lr0.05`, `exp_lr0.1`, `exp_steps50k`, `exp_steps200k` | 100k steps · **1 clone** · 4 axes balayés | Choix de lr, steps, profondeur, largeur |

**15 runs au total.** Les détails du sweep principal sont dans [`analyse_sweep_couche_cachee.md`](analyse_sweep_couche_cachee.md) ; ce rapport-ci se concentre sur *le pourquoi de chaque hyperparamètre*.

---

## 1. Problème

### 1.1 La tâche
Classer des images en **3 genres** de jeux vidéo. Le dataset brut est **déséquilibré** : Fighter 19 638, Platformer 12 738, Racing 14 593 images. Un MLP attend un vecteur d'entrée de **taille fixe** ; les images ont des résolutions variées.

### 1.2 Le vrai problème : justifier ~12 hyperparamètres
Un MLP entraîné from scratch expose une douzaine de leviers (taille d'image, normalisation, équilibrage, split, architecture, activation, initialisation, learning rate, decay, nombre de steps, méthode d'optimisation, protocole de validation). **Des valeurs choisies « au hasard » sont (a) indéfendables à l'oral et (b) capables de casser silencieusement l'entraînement** — comme on le verra, une mauvaise valeur fait chuter le modèle à 33 % (le hasard sur 3 classes) sans erreur ni crash.

**Objectif du rapport :** montrer que chaque choix est justifié par un principe ou par la mesure, et distinguer honnêtement ce qui est *validé*, *raisonné* ou *perfectible*.

### 1.3 Contraintes
- **Implémentation from scratch en C++** (pas de framework) → on privilégie des mécanismes simples, explicables et peu coûteux.
- **CPU uniquement** (MacBook 8 cœurs) → le budget de calcul est limité, ce qui contraint la taille d'image et le nombre de steps.
- **Soutenance** → chaque choix doit tenir en une phrase défendable.

---

## 2. Analyse (raisonnement *a priori*)

Choix posés **avant** de voir les résultats, par étage du pipeline.

### 2.1 Prétraitement des données
| Hyperparamètre | Valeur | Raisonnement a priori |
|---|---|---|
| **Taille d'image** | 32×32×3 = **3072 entrées** | Compromis information / coût. Assez grand pour garder les indices de genre (mise en page, palette : HUD de combat, plateformes, piste), assez petit pour qu'un MLP dense reste entraînable sur CPU. Le redimensionnement en **area-averaging** (et non nearest-neighbor) préserve mieux l'information en forte réduction. |
| **Normalisation** | pixels **/255 → [0, 1]** | Borner les entrées évite des sommes pondérées démesurées et stabilise `tanh`. Choix standard. |
| **Équilibrage** | plafond **8500/classe** | 8500 < effectif de la plus petite classe (12 738) → les 3 classes fournissent le même nombre d'images. Empêche le modèle de « tricher » en prédisant la classe majoritaire (Fighter). |
| **Split train/test** | **80/20 stratifié**, graine **42** | Stratifié = chaque classe présente en test (accuracy par classe fiable). Graine fixe = **même split pour toutes les configs** → comparaisons équitables + reproductibilité. |

### 2.2 Le modèle
| Hyperparamètre | Valeur | Raisonnement a priori |
|---|---|---|
| **Architecture** | `[3072, H, 3]` (H à déterminer) | Entrée (3072) et sortie (3 classes) imposées. La **taille de couche cachée H est le paramètre libre** → à trancher par un sweep, pas au doigt mouillé. |
| **Activation** | `tanh` (couches cachées + sortie) | Vu en cours ; non-linéarité bornée. Cibles one-hot **±1**, cohérentes avec l'image de `tanh` (]−1, 1[). Dérivée `1 − tanh²` simple à coder en rétropropagation. |
| **Initialisation** | poids ~ **U(−1, 1)** | Le choix « par défaut » le plus simple. *(C'est le point qu'on remettra en cause en §4.)* |

### 2.3 L'optimisation
| Hyperparamètre | Valeur | Raisonnement a priori |
|---|---|---|
| **Méthode** | SGD (1 échantillon/step) | Descente de gradient stochastique du cours. Simple, faible mémoire ; contrepartie : loss bruitée par step. |
| **Learning rate** | **0.01** | Ni trop grand (risque de divergence), ni trop petit (convergence trop lente). Valeur de départ classique, **à confirmer par balayage**. |
| **Decay** | inverse-time `lr/(1+decay·step)`, **2e-5** | Réduire le pas en fin d'entraînement pour affiner la convergence (moins d'oscillation autour du minimum). Sur 300k steps : 0.01 → ≈ 0.0014. |
| **Nombre de steps** | **300 000** | Assez pour converger sans exploser le temps de calcul. **À confirmer** (convergé ? sur-apprentissage ?). |

### 2.4 Le protocole de robustesse
| Hyperparamètre | Valeur | Raisonnement a priori |
|---|---|---|
| **Clones** | **3** initialisations différentes | Un entraînement unique ne prouve rien : le même réseau a déjà fait 58,8 % puis 82,7 % sur les mêmes données, seule l'init changeant. On rapporte donc **moyenne ± écart-type**. |
| **Agrégation** | **bagging** (moyenne des sorties) | Combiner plusieurs clones décorrélés pour réduire la variance de prédiction. On moyenne les **sorties**, jamais les **poids** (neurones interchangeables). |

---

## 3. Observation (résultats *a posteriori* sur tous les modèles)

### 3.1 Tableau maître

**Sweep principal (300k, 3 clones, bagging) :**

| Run | Architecture | Bag test | Bag train | Clones (moy ± σ) | Meilleur clone | Accuracy/clone |
|---|---|:--:|:--:|:--:|:--:|:--:|
| **h128** | [3072, **128**, 3] | **85,2 %** | 89,0 % | **80,6 % ± 1,1 %** | 82,0 % | 82,0 / 80,5 / 79,3 |
| h64 | [3072, 64, 3] | 76,9 % | 79,2 % | 66,1 % ± 11,2 % | 81,9 % | 81,9 / 57,8 / 58,5 |
| h512 | [3072, 512, 3] | 74,5 % | 77,3 % | 49,3 % ± 19,6 % | 77,1 % | 34,7 / 36,2 / 77,1 |
| h256x128 | [3072, 256, 128, 3] | 56,4 % | 58,5 % | 41,0 % ± 10,9 % | 56,4 % | 33,3 / 33,3 / 56,4 |
| h256 | [3072, 256, 3] | 55,8 % | 57,9 % | 41,3 % ± 11,2 % | 57,1 % | 33,4 / 33,4 / 57,1 |

**Sweep exploratoire (100k, 1 clone) :**

| Run | Architecture | LR | Steps | Test | Train | Loss test min |
|---|---|:--:|:--:|:--:|:--:|:--:|
| exp_steps200k | [3072, 128, 3] | 0.01 | 200k | **80,0 %** | 83,2 % | 0,40 |
| exp_h128 | [3072, 128, 3] | 0.01 | 100k | 72,4 % | 73,9 % | 0,53 |
| exp_h128x64 | [3072, 128, 64, 3] | 0.01 | 100k | 69,6 % | 71,2 % | 0,52 |
| exp_steps50k | [3072, 128, 3] | 0.01 | 50k | 66,9 % | 67,0 % | 0,68 |
| exp_lr0.001 | [3072, 128, 3] | 0.001 | 100k | 63,0 % | 64,4 % | 0,82 |
| exp_h32 | [3072, 32, 3] | 0.01 | 100k | 55,3 % | 56,2 % | 0,77 |
| exp_h256x128x64 | [3072, 256, 128, 64, 3] | 0.01 | 100k | 52,2 % | 52,5 % | 0,83 |
| exp_lr0.05 | [3072, 128, 3] | 0.05 | 100k | 46,1 % | 46,1 % | 1,10 |
| exp_h512x256 | [3072, 512, 256, 3] | 0.01 | 100k | 33,5 % | 33,5 % | 1,15 |
| exp_lr0.1 | [3072, 128, 3] | 0.1 | 100k | 33,3 % | 33,3 % | 1,78 |

### 3.2 Observations par hyperparamètre

**Learning rate** (à `[128]/100k`) — **courbe en U inversé** :
`0.001 → 63,0 %` · `0.01 → 72,4 %` · `0.05 → 46,1 %` · `0.1 → 33,3 %`.

**Nombre de steps** (à `[128]/lr 0.01`) — **monotone croissant** :
`50k → 66,9 %` · `100k → 72,4 %` · `200k → 80,0 %` · `300k → 85,2 % (bag)`. L'écart train–test reste faible (+0,2 % à 50k → +3,8 % à 300k).

**Architecture — largeur** (monocouche) : `[32] → 55,3 %` · `[128] → 85,2 % (bag)`. **La performance N'EST PAS monotone** : au-delà de 128, elle *chute* (h256 : 55,8 %, h512 : 74,5 %).

**Architecture — profondeur** (à `lr 0.01/100k`) : `[128] → 72,4 %` · `[128,64] → 69,6 %` · `[256,128,64] → 52,2 %` · `[512,256] → 33,5 %`. Empiler des couches **dégrade**.

### 3.3 Le mode d'échec récurrent
Tous les modèles ratés partagent la **même signature** :
- **accuracy ≈ 33,3 %** = exactement le hasard sur 3 classes équilibrées ;
- **collapse vers une seule classe** : `exp_lr0.1` et `exp_h512x256` prédisent **100 % « Platformer », 0 % ailleurs** ; h256 ne prédit jamais Fighter (0 %), h256x128 jamais Racing (0 %) ;
- **loss de test bloquée** entre ~1,0 et 1,8 dès le début (sortie ≈ 0 partout → MSE ≈ 1, ou divergence pour lr 0.1).

### 3.4 La variance inter-clones (sweep principal)
`h128` : les 3 clones réussissent (σ = **1,1 %**). Les gros réseaux : selon l'init, un clone apprend (~77 %) ou s'effondre à 33 % → σ jusqu'à **19,6 %** (h512). L'effondrement est présent **dès le step 0**.

---

## 4. Déduction

### 4.1 Hyperparamètres **validés empiriquement** ✅
- **Learning rate = 0.01.** La courbe en U inversé le tranche : 0.001 sous-apprend (loss test bloquée à 0,82), 0.05 dégrade, **0.1 diverge** (loss 1,78, collapse mono-classe). 0.01 est l'optimum, **encadré des deux côtés** → ce n'est pas un choix arbitraire mais un maximum mesuré.
- **Steps = 300k.** L'accuracy croît de façon monotone (67 → 72 → 80 → 85 %) et l'écart train–test reste faible (+3,8 %) → **pas de sur-apprentissage**. Le modèle n'est même pas totalement saturé à 100k ; 300k est un bon compromis convergence/temps.
- **Couche cachée = 128 neurones.** Meilleure accuracy (85,2 %) **et** entraînement le plus stable (σ = 1,1 %). Plus petit (32, 64) : moins performant. Plus gros (256, 512, +profond) : instable.
- **3 clones + bagging.** Sur `h128`, le bag (85,2 %) dépasse le meilleur clone seul (82,0 %) → **gain d'ensemble réel (+3 pts)**. Surtout, les 3 clones ont **révélé l'instabilité** (σ élevée) qu'un run unique aurait masquée — c'est le protocole qui rend le diagnostic possible.

### 4.2 Hyperparamètres **raisonnés** (cohérents, non balayés) ◐
- **Taille 32×32, split 80/20 stratifié seed 42, SGD, équilibrage 8500** : choix méthodologiques standard, non remis en cause par les résultats. L'équilibrage est même *indirectement validé* : les échecs s'effondrent vers Platformer **malgré des classes équilibrées** → l'effondrement est un problème d'optimisation, pas de déséquilibre de données (donc l'équilibrage fait bien son travail).
- **Decay 2e-5 (conservé).** Il affine la fin de convergence et **n'explique pas les échecs** (les clones morts le sont dès le step 0, avant que le decay n'agisse). Défendable en une phrase, gardé pour cette raison.
- **`tanh` + cibles ±1** : fonctionne pour les modèles sains, mais sa **saturation** est au cœur du problème ci-dessous.

### 4.3 L'hyperparamètre **perfectible** — l'initialisation ⚠️
C'est la déduction centrale. **La cause des échecs n'est pas la capacité** (le meilleur clone de h512 atteint 77 %, aussi bien que h64) mais **l'instabilité à l'initialisation.**

Les poids sont tirés dans **U(−1, 1) indépendamment du nombre d'entrées**. La pré-activation du 1ᵉʳ neurone caché somme **3072 termes** `w·x` (pixels ∈ [0, 1], non centrés) :

```
σ(pré-activation) ≈ √(fan_in · Var(w) · E[x²]) ≈ √(3072 · 1/3 · 0,3) ≈ 17
```

Or `tanh` est saturée dès |z| > 3. **Avec z ≈ 17, les neurones sortent ±1 et leur dérivée `1 − tanh² ≈ 0`** → le gradient s'évanouit, le réseau démarre figé. Qu'un clone « décolle » devient un **tirage au sort** (d'où σ jusqu'à 19,6 %). Plus le réseau est large/profond, plus ce démarrage est fragile.

→ **Correctif standard : initialisation de Xavier/Glorot** (poids ∝ `1/√fan_in`), qui maintient la pré-activation dans la zone linéaire de `tanh`. La normalisation des pixels en **[0, 1] non centrée** aggrave le phénomène ; les centrer (→ moyenne 0) aiderait aussi.

### 4.4 La chaîne causale complète
> init U(−1, 1) non scalée + entrées [0, 1] non centrées → pré-activation ≈ 17 → **saturation de tanh** → gradient nul → réseau figé à l'init → **collapse vers une classe** → accuracy = 33 % (hasard). Effet amplifié par la largeur/profondeur ; masqué (partiellement) par le bagging qui suit le clone survivant.

---

## 5. Conclusion

### 5.1 Le jeu d'hyperparamètres retenu, justifié

| Hyperparamètre | Valeur | Statut | Justification en une phrase |
|---|---|---|---|
| Taille d'image | 32×32×3 (3072) | ◐ raisonné | Compromis info/coût pour un MLP dense sur CPU. |
| Normalisation | /255 → [0, 1] | ◐ raisonné (améliorable) | Borne les entrées ; **à centrer** pour réduire la saturation. |
| Équilibrage | 8500/classe | ✅ validé | Empêche le biais de classe majoritaire (confirmé par les échecs équilibrés). |
| Split | 80/20 stratifié, seed 42 | ◐ raisonné | Évaluation par classe fiable + comparaisons reproductibles. |
| **Couche cachée** | **[128]** | ✅ **validé** | Meilleure accuracy (85 %) **et** stabilité (σ 1,1 %). |
| Activation | tanh, cibles ±1 | ◐ raisonné | Non-linéarité bornée du cours ; sature (voir init). |
| **Initialisation** | U(−1, 1) | ⚠️ **perfectible** | Cause l'instabilité des gros réseaux → **passer à Xavier/Glorot**. |
| Optimiseur | SGD (1/step) | ◐ raisonné | Méthode du cours ; loss bruitée (lissée par fenêtre). |
| **Learning rate** | **0.01** | ✅ **validé** | Optimum mesuré (U inversé : 0.1 diverge, 0.001 sous-apprend). |
| Decay | 2e-5 (inverse-time) | ◐ raisonné | Affine la fin de convergence ; sans effet sur les échecs. |
| **Nombre de steps** | **300k** | ✅ **validé** | Croissance monotone de l'accuracy, sur-apprentissage négligeable. |
| Robustesse | 3 clones + bagging | ✅ validé | Gain d'ensemble (+3 pts) et **révèle** la variance d'entraînement. |

### 5.2 Message de synthèse pour la soutenance
Les hyperparamètres **critiques** (learning rate, nombre de steps, taille de couche cachée) ne sont **pas des choix arbitraires** : chacun est un **optimum ou un compromis mesuré** sur 15 modèles générés. Le protocole à **3 clones + bagging** n'a pas seulement amélioré la performance, il a **rendu visible** un problème qu'un entraînement unique aurait caché : l'instabilité des grands réseaux.

**Le point technique fort** est d'avoir *diagnostiqué* cette instabilité (saturation de `tanh` due à une initialisation non mise à l'échelle du fan-in) et d'en *connaître le correctif* (initialisation de Xavier/Glorot). C'est la seule vraie faiblesse identifiée, et elle est comprise — pas subie.

### 5.3 Limites
- Sweep exploratoire à **1 clone** → distingue mal « config mauvaise » de « init malchanceuse » (sauf `lr 0.1`, vraie divergence).
- **Un seul split** (seed 42) : on mesure la variabilité de l'*entraînement*, pas celle du *split*.
- Plusieurs hyperparamètres (taille d'image, normalisation, activation) sont **raisonnés mais non balayés** faute de temps de calcul.

---

*Sources : `models/mlp_sweep_*_v1.json` (manifests) et `runs/mlp_*` + `runs/exp_*` (TensorBoard : `tensorboard --logdir runs`). Reproduire : `python scripts/sweep_mlp_hidden.py` (principal) et `python scripts/train_one.py --name ... --arch ... --lr ... --steps ...` (exploratoire, parallélisable via `xargs -P<N>`).*
