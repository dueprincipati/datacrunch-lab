import pandas as pd
import os
from src.features import preprocess_features, get_feature_names
from src.model import NTargetModel

# Configurazione
TARGETS = ["target_w", "target_r", "target_g", "target_b"] # Lista dei target N
MODEL_DIR = "resources"

def train(X_train: pd.DataFrame, y_train: pd.DataFrame, model_directory_path: str = MODEL_DIR):
    print("Inizio training N-Target...")
    
    # 1. Preprocessing
    feature_names = get_feature_names(X_train)
    X_processed = preprocess_features(X_train, feature_names)
    
    # 2. Inizializzazione e Training Modello
    # Puoi cambiare LinearRegression con XGBoost o altro qui facilmente
    model = NTargetModel(target_names=TARGETS)
    model.train(X_processed, y_train)
    
    # 3. Salvataggio
    os.makedirs(model_directory_path, exist_ok=True)
    model.save(os.path.join(model_directory_path, "n_target_model.joblib"))
    print("Training completato.")

def infer(X_test: pd.DataFrame, model_directory_path: str = MODEL_DIR):
    print("Inizio inferenza...")
    
    # 1. Caricamento Modello
    model = NTargetModel(target_names=TARGETS)
    model.load(os.path.join(model_directory_path, "n_target_model.joblib"))
    
    # 2. Preprocessing
    feature_names = get_feature_names(X_test)
    X_processed = preprocess_features(X_test, feature_names)
    
    # 3. Predizione
    preds = model.predict(X_processed)
    
    return pd.DataFrame({
        "date": X_test["date"], # O 'moon' a seconda della versione del dataset
        "id": X_test["id"],
        "prediction": preds
    })