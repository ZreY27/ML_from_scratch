# Machine Learning Framework from scratch

# Prérequis

- Version Python et compilateur C/C++ compatible
(Perso : Python 3.10 & gcc version 15.2.0)

# Extension requis

- CMakeTools
- C/C++
- Python

# Setup son environnement d'exécution
```
python -m venv nom_du_venv
```

# Import
```
pip install -r requirements.txt
```

utilisation de Vapnik (pour voir si notre modèle va bien généraliser)
il faut à peu près 10x le nombre d'élément dans le dataset par rapport aux nombres de paramètres.

(pixel x pixel x couleur) 

Si on utilise 32x32x1 : 1 024 paramètres donc 10 240 iamges.
Si on utilise 32x32x3 : 3 072 paramètres donc 30 720 images.
Si on utilise 64x64x1 : 4 096 paramètres donc 40 960 iamges.
Si on utilise 64x64x3 : 12 288 paramètres donc 122 880 images.

C'est pour calculer l'erreur de toutes les couches
et du coup quand on va faire la descente de gradient : elle va passer dans le tableau des erreurs (fourni par le rétropropagation) et on va corriger les erreurs.

No free lunch theorem