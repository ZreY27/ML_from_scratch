# apps/main.py — Serveur FastAPI : héberge les modèles pré-entraînés (C++)
# et expose /predict pour l'application cliente (index.html).
# Auteur : Antoine (application client/serveur).
import os
import sys
import tempfile
import shutil

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates

# Chemins absolus -> l'app est lançable quel que soit le répertoire courant
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)  # pour importer model_registry (situé à la racine)

# Autorise Python à charger les DLLs du runtime C++ sous Windows.
# On réutilise la logique partagée de scripts/training_utils.py (liste de
# dossiers candidats + variable d'environnement ML_ESGI_DLL_DIR).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import training_utils
training_utils.enable_cpp_dlls()

import ML_ESGI
import model_registry as registry

app = FastAPI()
templates = Jinja2Templates(directory=APP_DIR)

# Cache des modèles déjà chargés en mémoire : clé = (id, version)
_predictors = {}


def get_predictor(manifest):
    """Renvoie un Predictor (chargé une seule fois par couple id/version)."""
    key = (manifest["id"], manifest["version"])
    if key not in _predictors:
        _predictors[key] = registry.load_predictor(manifest, MODELS_DIR)
    return _predictors[key]


# Libellé d'algo pour le 1er sélecteur : base_type porte l'algo réel
# (mlp/rbf/svm/linear) ; les modèles simples n'ont que 'type'.
ALGO_LABELS = {"mlp": "MLP", "rbf": "RBF", "svm": "SVM", "linear": "Perceptron (linéaire)"}


def family_of(manifest):
    key = manifest.get("base_type") or manifest.get("type")
    return ALGO_LABELS.get(key, key)


def models_for_template():
    """Modèles disponibles (groupés par id), chacun avec sa famille et ses versions.

    Le 1er sélecteur de l'UI liste les FAMILLES, le 2e les modèles/versions de la
    famille choisie — d'où le champ 'family' ajouté ici.
    """
    models = []
    for mid, versions in sorted(registry.list_models(MODELS_DIR).items()):
        most_recent = versions[-1]
        models.append({
            "id": mid,
            "name": most_recent["name"],
            "type": most_recent["type"],
            "family": family_of(most_recent),
            "accuracy": (most_recent.get("metrics") or {}).get("accuracy"),
            "versions": [
                {
                    "version": m["version"],
                    "created": m.get("created", ""),
                    "width": m["width"],
                    "height": m["height"],
                    "name": m.get("name", most_recent["name"]),
                    # accuracy (test) si présente dans les métriques du manifeste, sinon None
                    "accuracy": (m.get("metrics") or {}).get("accuracy"),
                }
                # versions de la plus récente à la plus ancienne
                for m in sorted(versions, key=lambda x: x["version"], reverse=True)
            ],
        })
    return models


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "models": models_for_template()},
    )


@app.post("/predict")
async def predict(model_id: str = Form(...), version: str = Form(...), file: UploadFile = File(...)):
    versions = registry.list_models(MODELS_DIR).get(model_id)
    if not versions:
        return {"error": f"Modèle '{model_id}' introuvable."}

    # Version demandée (sinon : la plus récente)
    manifest = next((m for m in versions if str(m["version"]) == str(version)), versions[-1])

    # Sauvegarde temporaire de l'image pour que le C++ puisse la lire
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        img_data = ML_ESGI.load_and_resize_image(tmp_path, manifest["width"], manifest["height"])
        prediction = get_predictor(manifest).predict(img_data)
        return {"prediction": prediction, "model": manifest["name"], "version": manifest["version"]}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)
