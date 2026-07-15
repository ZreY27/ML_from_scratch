# Analyse du sweep MLP — balayage de la taille de couche cachée

*Partie individuelle : Perceptron Multi-Couches (PMC / MLP). Dataset : classification d'images de jeux vidéo en 3 genres (Fighter, Platformer, Racing).*

---

## 1. Approche

### 1.1 Objectif
Mesurer l'effet de **l'architecture de la couche cachée** (largeur et profondeur) sur la qualité de classification, à protocole d'entraînement fixe. Cinq configurations ont été balayées :

| Run | Architecture complète | Couche(s) cachée(s) |
|-----|----------------------|---------------------|
| `h64` | `[3072, 64, 3]` | 1 couche, 64 neurones |
| `h128` | `[3072, 128, 3]` | 1 couche, 128 neurones |
| `h256` | `[3072, 256, 3]` | 1 couche, 256 neurones |
| `h512` | `[3072, 512, 3]` | 1 couche, 512 neurones |
| `h256x128` | `[3072, 256, 128, 3]` | 2 couches (256 puis 128) |

La couche d'entrée (3072 = 32×32×3) et la couche de sortie (3 classes) sont fixes.

### 1.2 Protocole d'entraînement (identique pour les 5 runs)
- **Données** : 3 classes équilibrées à **8500 images/classe** (plafond), redimensionnées en 32×32, pixels normalisés dans **[0, 1]** (`ImageLoader.cpp` : `v /= 255.0`).
- **Split** : train/test stratifié 80/20, **graine fixe 42** → même partition pour toutes les configs (on mesure l'effet de l'architecture, pas du hasard du split).
- **Optimiseur** : descente de gradient stochastique (SGD), 1 échantillon/step.
- **Activation** : `tanh` sur les couches cachées **et** de sortie (mode classification), cibles one-hot **±1**.
- **Learning rate** : 0.01, avec **decay temporel inverse** `lr_t = lr / (1 + decay·step)`, `decay = 2e-5`. Sur 300k steps, le LR décroît de 0.01 → ≈ 0.0014 (÷7) — il affine la convergence en fin d'entraînement.
- **Durée** : **300 000 steps** par clone.
- **Initialisation des poids** : uniforme **U(−1, 1)**.

### 1.3 Protocole de robustesse (le point clé de la méthodo)
Un entraînement unique ne prouve rien : notre MLP a déjà produit 58,8 % puis 82,7 % sur les **mêmes** données, seule l'initialisation aléatoire changeant. On entraîne donc **3 clones** par configuration (3 initialisations différentes) et on rapporte :
- la **moyenne ± écart-type** des accuracies des clones (résultat robuste) ;
- le **meilleur clone** ;
- le **bagging** des 3 clones (moyenne des vecteurs de sortie puis `argmax`) — c'est le modèle « déployé ».

On moyenne des **mesures** et des **sorties**, jamais les **poids** (les neurones cachés d'un MLP sont interchangeables : moyenner les poids de deux réseaux corrects peut donner un réseau cassé).

Une **courbe de loss de test** (MSE moyenne) est mesurée tous les 3000 steps pendant l'entraînement, alignée sur la courbe de train → visualisation directe convergence / sur-apprentissage dans TensorBoard (`tensorboard --logdir runs`).

> ⚠️ **Note d'honnêteté** — nombre de clones : les runs actuels utilisent **3 clones** (voir `n_clones: 3` dans chaque manifest, 3 courbes `clone_*` dans TensorBoard). Des fichiers de poids `..._var4.txt` / `..._var5.txt` traînent pour `h64` et `h128` : ce sont des **vestiges d'un run antérieur à 5 clones**, non référencés par les manifests `v1`. Toute l'analyse ci-dessous porte sur les 3 clones effectivement loggés.

---

## 2. Résultats

### 2.1 Tableau de synthèse (bagging = modèle déployé)

| Config | Bag **test** | Bag train | Écart (sur-app.) | Clones (moy ± σ) | Meilleur clone |
|--------|:---:|:---:|:---:|:---:|:---:|
| **h128** | **85,2 %** | 89,0 % | +3,8 % | **80,6 % ± 1,1 %** | 82,0 % |
| h64 | 76,9 % | 79,2 % | +2,3 % | 66,1 % ± 11,2 % | 81,9 % |
| h512 | 74,5 % | 77,3 % | +2,8 % | 49,3 % ± 19,6 % | 77,1 % |
| h256x128 | 56,4 % | 58,5 % | +2,1 % | 41,0 % ± 10,9 % | 56,4 % |
| h256 | 55,8 % | 57,9 % | +2,1 % | 41,3 % ± 11,2 % | 57,1 % |

*(trié par accuracy test décroissante)*

### 2.2 Accuracy par clone — la « variance » de l'entraînement

C'est la donnée la plus instructive du sweep :

| Config | Clone 1 | Clone 2 | Clone 3 | Lecture |
|--------|:---:|:---:|:---:|---|
| **h128** | 82,0 % | 80,5 % | 79,3 % | **les 3 clones réussissent** → σ minuscule (1,1 %) |
| h64 | 81,9 % | 57,8 % | 58,5 % | 1 bon, 2 médiocres → σ élevée |
| h256 | **33,4 %** | **33,4 %** | 57,1 % | **2 clones à ≈ 1/3 = hasard** |
| h512 | **34,7 %** | **36,2 %** | 77,1 % | 2 clones effondrés, 1 excellent |
| h256x128 | **33,3 %** | **33,3 %** | 56,4 % | **2 clones à exactement 1/3** |

Sur 3 classes équilibrées, **33,3 % = le hasard** : ces clones-là n'ont rien appris.

### 2.3 Preuve de l'effondrement : accuracy par classe (bagging)

| Config | Fighter | Platformer | Racing | Symptôme |
|--------|:---:|:---:|:---:|---|
| h128 | 86,8 % | 89,6 % | 79,2 % | équilibré, sain |
| h64 | 87,2 % | 92,8 % | 50,6 % | Racing plus faible |
| h256 | **0,1 %** | 97,4 % | 69,9 % | **Fighter jamais prédit** |
| h256x128 | 72,8 % | 96,4 % | **0,0 %** | **Racing jamais prédit** |
| h512 | 64,0 % | 92,7 % | 66,8 % | rattrapé par le clone vivant |

Quand un réseau s'effondre, il **collapse vers une classe** : il prédit presque toujours « Platformer » et ignore une autre classe (Fighter à 0,1 % pour h256, Racing à 0,0 % pour h256x128). Cohérent avec une accuracy ≈ 1/3.

### 2.4 Courbes de loss (TensorBoard)
- **Clones qui apprennent** (tous ceux de h128, clone_1 de h64, clone_3 de h256/h512/h256x128) : loss train qui descend de ~2,5 vers 0,3–0,8, loss test qui suit. **Minimum atteint tard (≈ step 279k–297k)** puis remontée très faible (+0,01 à +0,03) → **quasi-convergence, sur-apprentissage négligeable**.
- **Clones effondrés** : loss bloquée à **≈ 1,0** du début à la fin. Avec des sorties tanh proches de 0 et des cibles ±1, la MSE moyenne par sortie vaut exactement `(0−(±1))² = 1` → **la loss ≈ 1,0 est la signature d'un réseau figé qui ne produit rien**.

---

## 3. Analyse

### 3.1 La performance n'est PAS monotone avec la taille
On pourrait attendre « plus de neurones → meilleur ». C'est faux ici. L'ordre observé est :

> **h128 (85 %) > h64 (77 %) > h512 (75 %) ≫ h256x128 (56 %) ≈ h256 (56 %)**

`h128` est un **point optimal** : assez de capacité pour séparer les 3 genres, sans l'instabilité des réseaux plus gros. Doubler (h256) ou quadrupler (h512) la largeur **dégrade** le résultat, et ajouter une 2ᵉ couche (h256x128) ne fait qu'imiter l'échec de h256.

### 3.2 Le vrai problème : l'instabilité à l'initialisation, pas la capacité
La cause n'est pas un manque de puissance des gros réseaux (le meilleur clone de h512 atteint 77 %, aussi bien que h64). C'est que, au-delà de 128 neurones, **la réussite de l'entraînement devient un tirage au sort** : selon la graine d'initialisation, un clone apprend (~77 %) ou reste figé au hasard (~33 %). D'où l'explosion de l'écart-type (σ = 19,6 % pour h512 contre 1,1 % pour h128).

**Diagnostic — saturation de `tanh` due à l'initialisation.** Les poids sont tirés dans U(−1, 1), indépendamment du nombre d'entrées. La pré-activation du 1ᵉʳ neurone caché est une somme de **3072 termes** `w·x` (pixels x ∈ [0, 1]). Son écart-type vaut approximativement :

```
σ_pré-activation ≈ √(fan_in · Var(w) · E[x²]) ≈ √(3072 · 1/3 · 0,3) ≈ 17
```

Or `tanh` est déjà saturée (plateau ±1) dès |z| > 3. **Avec z ≈ 17, tous les neurones cachés sortent ±1**, et leur dérivée `1 − tanh² ≈ 0` : le gradient qui devrait remonter dans le réseau **s'évanouit**. Le réseau démarre quasi figé ; seuls les clones dont le tirage laisse par chance assez de neurones dans la zone linéaire parviennent à « décoller ». Plus le réseau est large/profond (plus de sommes larges, notamment en sortie où σ ≈ √(H/3) croît avec H), plus ce démarrage est fragile.

> C'est le problème classique que résout l'**initialisation de Xavier/Glorot** (mettre les poids à l'échelle `1/√fan_in`), absente ici. Explication simple et défendable en soutenance : *« nos poids ne sont pas mis à l'échelle du nombre d'entrées, donc tanh sature et l'apprentissage dépend de la chance du tirage — ce que corrigerait une init de Xavier »*.

### 3.3 Ce que révèle le bagging
Le bagging ne se comporte pas pareil selon le régime :

- **Régime stable (h128)** : les 3 clones sont bons **et** décorrélés → le bagging **apporte réellement** : 85,2 % contre 82,0 % pour le meilleur clone seul (+3,2 pts) et 80,6 % en moyenne. C'est le bénéfice d'ensemble attendu.
- **Régime « clones morts » (h256/h512/h256x128)** : les clones effondrés sortent ≈ 0, donc **ne pèsent presque rien** dans la moyenne des sorties → le `argmax` du bag **suit le clone vivant**. Résultat : **bag ≈ meilleur clone** (h512 : bag 74,5 % vs best 77,1 % ; h256x128 : bag 56,4 % = best 56,4 %), très au-dessus de la moyenne des clones. Le bagging « sauve » en ignorant les morts, mais ne crée pas de performance.
- **Régime intermédiaire (h64)** : 1 bon clone (82 %) + 2 médiocres mais vivants (58 %) → la moyenne **dilue** le bon → bag 76,9 % **en dessous** du meilleur clone (81,9 %).

**Le bagging n'a été bénéfique que dans le régime où tous les clones apprennent** (h128). Ailleurs, il masque l'instabilité plutôt qu'il ne l'améliore.

---

## 4. Conclusion

1. **`h128` (une couche de 128 neurones) est la meilleure architecture** de ce balayage : 85,2 % en test, entraînement stable (σ = 1,1 %), sur-apprentissage faible (+3,8 %). C'est la configuration à retenir pour le modèle déployé.
2. **Augmenter la taille dégrade la performance** ici — non par manque de capacité, mais par **instabilité d'entraînement** : au-delà de 128 neurones, une partie des clones s'effondre au niveau du hasard (33 %) et collapse vers une seule classe.
3. **Cause identifiée** : saturation de `tanh` provoquée par une initialisation U(−1, 1) non mise à l'échelle du fan-in (3072). La convergence devient dépendante du tirage aléatoire.
4. **Le bagging n'apporte de gain réel que lorsque tous les clones convergent** (cas h128, +3 pts). Face à des clones morts, il se contente de suivre le meilleur.

### Pistes d'amélioration (prochaines expériences)
- **Initialisation de Xavier/Glorot** (`w ~ U(−√(6/(n_in+n_out)), +√(...))`) : correctif direct de la saturation → devrait stabiliser h256/h512 et lever leur variance. *Priorité 1.*
- **Standardisation des entrées** (centrer-réduire les pixels autour de 0 plutôt que [0, 1]) : réduit encore la pré-activation moyenne.
- Balayer le **learning rate** et le **nombre de steps** (fait — voir §5) : confirme que **lr 0.01 est optimal** (0.1 diverge) et que **plus de steps améliore** (200k > 100k > 50k, sans sur-apprentissage).
- À terme, tester l'activation **ReLU** sur les couches cachées (pas de saturation bilatérale).

### Limites de l'étude
- **3 clones seulement** → l'écart-type est estimé sur peu de points (à interpréter qualitativement).
- **Un seul split** train/test (graine 42) : on mesure la variabilité de *l'entraînement*, pas celle du split.
- Le `decay` du learning rate est conservé (choix assumé, simple à expliquer) ; il n'explique pas l'effondrement (les clones morts le sont dès le step 0).

---

## 5. Sweep exploratoire complémentaire (profondeur × largeur × LR × epochs)

*Protocole **allégé et différent** du principal : **1 clone** (pas de bagging) et **100k steps** (sauf l'axe epochs), pour balayer vite **10 configurations en parallèle** (pool de 6 process, ~3 h). Les accuracies ne sont donc **PAS comparables** aux 85 % du §2 (bag de 3 clones à 300k) : on lit ici des **tendances par axe**, autour d'un pivot commun `[128] / lr 0.01 / 100k = 72,4 %`.*

### 5.1 Résultats (10 runs, triés par accuracy test)

| Run | Arch | LR | Steps | Test | Train | Écart | Temps |
|-----|------|:--:|:--:|:--:|:--:|:--:|:--:|
| exp_steps200k | [128] | 0.01 | 200k | **80,0 %** | 83,2 % | +3,2 % | 53 min |
| exp_h128 *(pivot)* | [128] | 0.01 | 100k | 72,4 % | 73,9 % | +1,5 % | 47 min |
| exp_h128x64 | [128, 64] | 0.01 | 100k | 69,6 % | 71,2 % | +1,6 % | 49 min |
| exp_steps50k | [128] | 0.01 | 50k | 66,9 % | 67,0 % | +0,2 % | 40 min |
| exp_lr0.001 | [128] | 0.001 | 100k | 63,0 % | 64,4 % | +1,4 % | 47 min |
| exp_h32 | [32] | 0.01 | 100k | 55,3 % | 56,2 % | +0,9 % | 18 min |
| exp_h256x128x64 | [256, 128, 64] | 0.01 | 100k | 52,2 % | 52,5 % | +0,2 % | 105 min |
| exp_lr0.05 | [128] | 0.05 | 100k | 46,1 % | 46,1 % | +0,1 % | 47 min |
| exp_h512x256 | [512, 256] | 0.01 | 100k | 33,5 % | 33,5 % | −0,1 % | 180 min |
| exp_lr0.1 | [128] | 0.1 | 100k | 33,3 % | 33,3 % | 0,0 % | 45 min |

### 5.2 Lecture par axe

- **Learning rate (courbe en U inversé — le résultat le plus net).** À `[128] / 100k` : `0.001 → 63,0 %` (trop bas, sous-apprentissage : loss test bloquée à 0,83) · `0.01 → 72,4 %` (**optimum**) · `0.05 → 46,1 %` (trop haut) · `0.1 → 33,3 %` (**divergence** : loss test 1,78, prédit uniquement « Platformer »). → **0.01 est le bon choix, encadré des deux côtés.**
- **Nombre de steps (monotone, sans sur-apprentissage).** À `[128] / lr 0.01` : `50k → 66,9 %` · `100k → 72,4 %` · `200k → 80,0 %`. L'écart train–test reste faible (+0,2 → +3,2 %). → **le modèle n'est pas encore convergé à 100k** ; cohérent avec le choix de 300k au §2.
- **Profondeur (plus profond = pire ici).** À `lr 0.01 / 100k` : `[128] → 72,4 %` · `[128,64] → 69,6 %` · `[256,128,64] → 52,2 %` · `[512,256] → 33,5 %` (effondré). → empiler des couches **dégrade**, comme au §2.
- **Largeur (monocouche).** `[32] → 55,3 %` (trop petit : « Racing » tombe à 0 %) · `[128] → 72,4 %`. → 32 neurones insuffisants ; 128 confirme son statut de bon compromis.

### 5.3 Confirmation transversale
La **signature d'effondrement** est identique au §2 : les configs ratées prédisent une seule classe (`exp_lr0.1` et `exp_h512x256` → 100 % « Platformer », 0 % ailleurs) et leur loss test reste bloquée à ~1,0–1,8. Cela **renforce le diagnostic de saturation / instabilité**.

> ⚠️ **Limite** : 1 seul clone par config → on ne peut pas distinguer « config mauvaise » de « init malchanceuse ». Vu la forte variance du §2, l'effondrement de `exp_h512x256` est peut-être en partie dû au tirage (son meilleur clone atteignait 77 % à 300k). En revanche, l'effondrement de `exp_lr0.1` est une **vraie divergence** (LR trop grand). Pour trancher fermement sur profondeur/largeur, rejouer ces configs en 3 clones.

---

*Reproduire : sweep principal `python scripts/sweep_mlp_hidden.py` ; sweep exploratoire `python scripts/train_one.py --name ... --arch ... --lr ... --steps ...` (une config ; parallélisable via `xargs -P<N>`, voir `revision/README.md`). Visualiser : `tensorboard --logdir runs`. Données sources : dossiers `runs/mlp_*` + `runs/exp_*` et manifests `models/mlp_sweep_*_v1.json`.*
