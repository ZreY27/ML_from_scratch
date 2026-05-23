import os
import json
import tempfile
import shutil
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates

# Autorise Python à charger les DLLs du compilateur C++ (MSYS2) sous Windows
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r"C:\msys64\ucrt64\bin")

import ML_ESGI

app = FastAPI()
templates = Jinja2Templates(directory="web/templates")

CONFIG_PATH = "web/config.json"
MODELS_DIR = "models"

# Cache pour stocker les modèles déjà chargés en mémoire (optimisation de vitesse)
loaded_models = {}

def get_models_config():
    """Charge la liste des modèles disponibles depuis le fichier JSON"""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def get_or_load_model(model_config):
    """Instancie et charge les poids du modèle C++ s'il n'est pas déjà en mémoire"""
    model_id = model_config["id"]
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    m_type = model_config["type"]
    filepath = os.path.join(MODELS_DIR, model_config["file"])
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier de poids {filepath} est introuvable.")

    if m_type == "mlp":
        model = ML_ESGI.MLP([1, 1], True) # Architecture bidon temporaire (écrasée par load())
        model.load(filepath)
    elif m_type == "linear":
        model = ML_ESGI.LinearModel(1, True) # Dimension bidon temporaire
        model.load(filepath)
    else:
        raise ValueError(f"Support du type de modèle '{m_type}' non implémenté.")
        
    loaded_models[model_id] = model
    return model

@app.get("/")
def read_root(request: Request):
    models = get_models_config()
    return templates.TemplateResponse("index.html", {"request": request, "models": models})

@app.post("/predict")
async def predict(model_id: str = Form(...), file: UploadFile = File(...)):
    models = get_models_config()
    model_config = next((m for m in models if m["id"] == model_id), None)
    
    if not model_config:
        return {"error": "Modèle introuvable dans la configuration."}
    
    # Sauvegarde temporaire de l'image pour que le C++ puisse la lire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        
    try:
        img_data = ML_ESGI.load_and_resize_image(tmp_path, model_config["width"], model_config["height"])
        model = get_or_load_model(model_config)
        
        prediction = model.predict(img_data)
        
        if model_config["type"] == "mlp":
            max_idx = max(range(len(prediction)), key=lambda i: prediction[i])
            return {"prediction": model_config["classes"][max_idx]}
        elif model_config["type"] == "linear":
            return {"prediction": model_config["classes"][0] if prediction > 0 else model_config["classes"][1]}
            
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)