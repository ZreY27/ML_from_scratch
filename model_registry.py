"""
model_registry.py — Gestion des modèles entraînés via des manifestes JSON.
Auteur : Antoine (interfaçage app / registre de modèles).

Idée : un fichier de poids (.txt) ne décrit pas le modèle (classes, taille image, type,
et pour le One-vs-Rest : quels fichiers vont ensemble). On pose donc à côté de chaque modèle
un petit manifeste `.json` qui porte ces métadonnées. L'app n'a plus qu'à scanner `models/*.json`
pour proposer automatiquement les modèles et leurs versions — sans config éditée à la main.

Convention de nommage : `<id>_v<N>.txt` (poids) + `<id>_v<N>.json` (manifeste).
La version N est un entier auto-incrémenté ; la date complète est gardée dans le manifeste.

Schéma d'un manifeste :
{
  "id": "linear_genres",            # identité logique (regroupe les versions)
  "name": "Perceptron - Genres",     # libellé affiché
  "version": 3,                       # entier, auto-incrémenté
  "created": "2026-06-28T15:30:00",
  "type": "onevsrest",              # "mlp" | "linear" | "svm" | "rbf" | "onevsrest"
  "base_type": "linear",            # (onevsrest seulement) type des sous-modèles binaires
  "width": 32, "height": 32,
  "classes": ["Fighter", "Racing", "Platformer"],
  "weights": "linear_genres_v3.txt"  # str (modèle simple) OU {classe: fichier} (onevsrest)
  "hyperparameters": { ... }          # optionnel : ce qu'on a RÉGLÉ (lr, epochs, decay, archi, C, max_per_class…)
  "metrics": { ... }                  # optionnel : ce qu'on a MESURÉ (accuracy, counts, final_loss…)
}

Garder `hyperparameters` (réglages) et `metrics` (résultats) séparés permet de tracer
l'évolution de l'accuracy en fonction des hyperparamètres ET du volume de dataset.
"""

import os
import json
import glob
import datetime

DEFAULT_MODELS_DIR = "models"


# --------------------------------------------------------------------------- #
#  Lecture / listing des manifestes                                           #
# --------------------------------------------------------------------------- #
def list_models(models_dir=DEFAULT_MODELS_DIR):
    """Scanne models_dir et retourne {id: [manifestes triés par version croissante]}.

    Chaque manifeste reçoit en plus une clé interne "_manifest_path".
    """
    groups = {}
    for path in glob.glob(os.path.join(models_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # fichier .json non-manifeste ou illisible : on ignore
        if "id" not in manifest or "version" not in manifest:
            continue
        manifest["_manifest_path"] = path
        groups.setdefault(manifest["id"], []).append(manifest)

    for mid in groups:
        groups[mid].sort(key=lambda m: m["version"])
    return groups


def latest(model_id, models_dir=DEFAULT_MODELS_DIR):
    """Retourne le manifeste de la version la plus récente d'un id, ou None."""
    versions = list_models(models_dir).get(model_id, [])
    return versions[-1] if versions else None


def next_version(model_id, models_dir=DEFAULT_MODELS_DIR):
    """Prochaine version à utiliser pour cet id (max existant + 1, sinon 1)."""
    versions = list_models(models_dir).get(model_id, [])
    return (versions[-1]["version"] + 1) if versions else 1


# --------------------------------------------------------------------------- #
#  Sauvegarde (poids + manifeste, version auto-incrémentée)                    #
# --------------------------------------------------------------------------- #
def _write_manifest(models_dir, model_id, version, manifest):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{model_id}_v{version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def save_single(model, model_id, name, model_type, classes, width, height,
                models_dir=DEFAULT_MODELS_DIR, metrics=None, hyperparams=None):
    """Sauve un modèle simple (mlp/linear/svm/rbf) + son manifeste.

    hyperparams : dict des réglages d'entraînement (lr, epochs, archi, max_per_class…), optionnel.
    Retourne (version, chemin_du_manifeste).
    """
    version = next_version(model_id, models_dir)
    weights_file = f"{model_id}_v{version}.txt"
    model.save(os.path.join(models_dir, weights_file))

    manifest = {
        "id": model_id, "name": name, "version": version, "created": _now(),
        "type": model_type, "width": width, "height": height,
        "classes": list(classes), "weights": weights_file,
    }
    if hyperparams:
        manifest["hyperparameters"] = hyperparams
    if metrics:
        manifest["metrics"] = metrics
    return version, _write_manifest(models_dir, model_id, version, manifest)


def save_onevsrest(models_by_class, model_id, name, base_type, width, height,
                   models_dir=DEFAULT_MODELS_DIR, metrics=None, hyperparams=None):
    """Sauve N modèles binaires (un par classe) regroupés en 1 classifieur One-vs-Rest.

    models_by_class : dict {nom_de_classe: modèle_binaire_entraîné} (ordre = ordre des classes).
    hyperparams : dict des réglages d'entraînement (lr, epochs, C, max_per_class…), optionnel.
    Retourne (version, chemin_du_manifeste).
    """
    version = next_version(model_id, models_dir)
    weights = {}
    for cls, model in models_by_class.items():
        safe = str(cls).lower().replace(" ", "_")
        weights_file = f"{model_id}_v{version}_{safe}.txt"
        model.save(os.path.join(models_dir, weights_file))
        weights[cls] = weights_file

    manifest = {
        "id": model_id, "name": name, "version": version, "created": _now(),
        "type": "onevsrest", "base_type": base_type,
        "width": width, "height": height,
        "classes": list(models_by_class.keys()), "weights": weights,
    }
    if hyperparams:
        manifest["hyperparameters"] = hyperparams
    if metrics:
        manifest["metrics"] = metrics
    return version, _write_manifest(models_dir, model_id, version, manifest)


def save_bag(variants, model_id, name, base_type, classes, width, height,
             models_dir=DEFAULT_MODELS_DIR, sub_base_type=None, metrics=None, hyperparams=None):
    """Sauve un ensemble (bagging) de N variants du même modèle + son manifeste.

    variants : liste de N modèles entraînés. Chaque variant est soit un modèle simple
    (mlp/rbf), soit un dict {classe: modèle_binaire} (One-vs-Rest linear/svm).
    À l'inférence, le Predictor moyennera les SORTIES des N variants (cf. Predictor.scores).

    base_type     : "mlp" | "rbf" | "onevsrest"
    sub_base_type : type des binaires si base_type == "onevsrest" ("linear" | "svm")
    Retourne (version, chemin_du_manifeste).
    """
    version = next_version(model_id, models_dir)
    weights = []
    for k, variant in enumerate(variants, start=1):
        if isinstance(variant, dict):  # One-vs-Rest : un fichier par classe
            entry = {}
            for cls, model in variant.items():
                safe = str(cls).lower().replace(" ", "_")
                f = f"{model_id}_v{version}_var{k}_{safe}.txt"
                model.save(os.path.join(models_dir, f))
                entry[cls] = f
            weights.append(entry)
        else:                          # modèle simple : un fichier par variant
            f = f"{model_id}_v{version}_var{k}.txt"
            variant.save(os.path.join(models_dir, f))
            weights.append(f)

    manifest = {
        "id": model_id, "name": name, "version": version, "created": _now(),
        "type": "bag", "base_type": base_type,
        "width": width, "height": height,
        "classes": list(classes), "weights": weights,
    }
    if sub_base_type:
        manifest["sub_base_type"] = sub_base_type
    if hyperparams:
        manifest["hyperparameters"] = hyperparams
    if metrics:
        manifest["metrics"] = metrics
    return version, _write_manifest(models_dir, model_id, version, manifest)


def register_existing(model_id, name, model_type, classes, width, height, weights,
                      models_dir=DEFAULT_MODELS_DIR, base_type=None, version=None,
                      metrics=None, hyperparams=None):
    """Crée un manifeste pour des fichiers de poids DÉJÀ présents (sans model.save()).

    weights : nom de fichier (modèle simple) ou dict {classe: fichier} (onevsrest).
    Sert à enregistrer des modèles entraînés avant l'arrivée des manifestes.
    Retourne (version, chemin_du_manifeste).
    """
    if version is None:
        version = next_version(model_id, models_dir)
    manifest = {
        "id": model_id, "name": name, "version": version, "created": _now(),
        "type": model_type, "width": width, "height": height,
        "classes": list(classes), "weights": weights,
    }
    if base_type:
        manifest["base_type"] = base_type
    if hyperparams:
        manifest["hyperparameters"] = hyperparams
    if metrics:
        manifest["metrics"] = metrics
    return version, _write_manifest(models_dir, model_id, version, manifest)


# --------------------------------------------------------------------------- #
#  Chargement / inférence                                                      #
# --------------------------------------------------------------------------- #
def _new_model(ml, base_type):
    """Instancie un modèle vide (dimensions bidons : écrasées par load())."""
    if base_type == "linear":
        return ml.LinearModel(1, True)
    if base_type == "svm":
        return ml.SVM(1)
    if base_type == "mlp":
        return ml.MLP([1, 1], True)
    if base_type == "rbf":
        return ml.RBF(1, 1, 1)
    raise ValueError(f"Type de modèle inconnu : {base_type!r}")


class Predictor:
    """Modèle prêt à prédire, construit depuis un manifeste.

    .predict(x) prend un vecteur d'entrée aplati (ex: image 32x32x3 -> 3072 floats)
    et retourne le NOM de la classe prédite.
    """

    def __init__(self, manifest, model=None, sub_models=None, variants=None):
        self.manifest = manifest
        self.model = model            # modèle simple
        self.sub_models = sub_models  # One-vs-Rest : liste alignée sur classes
        self.variants = variants      # bag : liste de Predictor (un par variant)

    @property
    def classes(self):
        return self.manifest["classes"]

    def scores(self, x):
        """Vecteur de scores bruts, un par classe (avant argmax).

        C'est la brique du bagging : pour un "bag", on moyenne les vecteurs de
        scores des N variants (moyenne des SORTIES, jamais des poids — comme le
        `bag_y_pred = np.mean(folds_y_pred)` du cours), puis predict() fait l'argmax.
        """
        t = self.manifest["type"]
        if t == "bag":
            vecteurs = [v.scores(x) for v in self.variants]
            n = len(vecteurs)
            return [sum(v[i] for v in vecteurs) / n for i in range(len(vecteurs[0]))]
        if t == "onevsrest":
            return [m.predict_raw(x) for m in self.sub_models]
        if t == "rbf":
            return list(self.model.predict_raw(x))
        if t == "mlp":
            return list(self.model.predict(x))  # sorties tanh brutes = scores
        raise ValueError(f"scores() non défini pour le type {t!r}")

    def predict(self, x):
        classes = self.classes

        # Types multi-classes (bag, One-vs-Rest, mlp, rbf) : argmax des scores
        if self.manifest["type"] in ("bag", "onevsrest", "mlp", "rbf"):
            s = self.scores(x)
            if len(s) > 1:
                return classes[s.index(max(s))]
            return classes[0] if s[0] >= 0 else classes[1]  # binaire à 1 sortie

        # Modèle simple binaire (linear/svm) : signe de la sortie
        out = self.model.predict(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return classes[0] if out >= 0 else classes[1]


def load_predictor(manifest, models_dir=DEFAULT_MODELS_DIR):
    """Construit un Predictor depuis un manifeste (charge les poids C++)."""
    import ML_ESGI as ml  # import tardif : suppose os.add_dll_directory déjà fait côté appelant

    if manifest["type"] == "bag":
        # Un bag = N variants ; chaque variant devient un Predictor interne,
        # et le Predictor "bag" moyenne leurs scores à l'inférence.
        variants = []
        for entry in manifest["weights"]:
            if isinstance(entry, dict):  # variant One-vs-Rest
                subs = []
                for cls in manifest["classes"]:
                    model = _new_model(ml, manifest["sub_base_type"])
                    model.load(os.path.join(models_dir, entry[cls]))
                    subs.append(model)
                variants.append(Predictor({"type": "onevsrest", "classes": manifest["classes"]},
                                          sub_models=subs))
            else:                        # variant simple (mlp/rbf)
                model = _new_model(ml, manifest["base_type"])
                model.load(os.path.join(models_dir, entry))
                variants.append(Predictor({"type": manifest["base_type"], "classes": manifest["classes"]},
                                          model=model))
        return Predictor(manifest, variants=variants)

    if manifest["type"] == "onevsrest":
        subs = []
        for cls in manifest["classes"]:
            model = _new_model(ml, manifest["base_type"])
            model.load(os.path.join(models_dir, manifest["weights"][cls]))
            subs.append(model)
        return Predictor(manifest, sub_models=subs)

    model = _new_model(ml, manifest["type"])
    model.load(os.path.join(models_dir, manifest["weights"]))
    return Predictor(manifest, model=model)
