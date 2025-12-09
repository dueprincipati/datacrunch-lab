import pandas as pd
import os
import joblib
from src.features import preprocess_features, get_feature_names
from src.model import NTargetModel

# Configurazione: I 4 target di DataCrunch
TARGETS = ["target_w", "target_r", "target_g", "target_b"]
MODEL_DIR = "resources"

def train(X_train: pd.DataFrame, y_train: pd.DataFrame, model_directory_path: str = MODEL_DIR):
    """Funzione chiamata dalla piattaforma per il training."""
    print("Inizio training N-Target...")
    
    # 1. Preprocessing
    feature_names = get_feature_names(X_train)
    X_processed = preprocess_features(X_train, feature_names)
    
    # 2. Training
    model = NTargetModel(target_names=TARGETS)
    model.train(X_processed, y_train)
    
    # 3. Salvataggio
    os.makedirs(model_directory_path, exist_ok=True)
    model.save(model_directory_path)
    print("Training completato e modello salvato.")

def infer(X_test: pd.DataFrame, model_directory_path: str = MODEL_DIR):
    """Funzione chiamata dalla piattaforma per l'inferenza."""
    
    # 1. Caricamento
    model = NTargetModel(target_names=TARGETS)
    model.load(model_directory_path)
    
    # 2. Preprocessing
    feature_names = get_feature_names(X_test)
    X_processed = preprocess_features(X_test, feature_names)
    
    # 3. Predizione
    preds = model.predict(X_processed)
    
    # 4. Gestione dinamica colonna temporale (FIX per KeyError: 'date')
    # Controlliamo se nel dataset c'è 'date' o 'moon' e usiamo quella presente
    time_col = "date" if "date" in X_test.columns else "moon"
    
    # 5. Formattazione output
    return pd.DataFrame({
        time_col: X_test[time_col],
        "id": X_test["id"],
        "prediction": preds
    })