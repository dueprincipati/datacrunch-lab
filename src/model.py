import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression

class NTargetModel:
    def __init__(self, target_names, model_class=LinearRegression, model_params={}):
        self.target_names = target_names
        self.models = {}
        self.model_class = model_class
        self.model_params = model_params

    def train(self, X, y):
        """Addestra un modello indipendente per ogni target."""
        for target in self.target_names:
            if target in y.columns:
                print(f"Training model for {target}...")
                model = self.model_class(**self.model_params)
                # Filtra righe dove il target è NaN
                valid_mask = y[target].notna()
                model.fit(X[valid_mask], y.loc[valid_mask, target])
                self.models[target] = model
            else:
                print(f"Warning: {target} not found in training labels.")

    def predict(self, X):
        """Fa predizioni per tutti i target e restituisce la media (ensemble semplice)."""
        predictions = pd.DataFrame(index=X.index)
        
        for target, model in self.models.items():
            predictions[target] = model.predict(X)
            
        # Qui implementiamo la logica 'Ensemble': Media di tutti i target
        # Spesso predire più target aiuta a stabilizzare il segnale
        return predictions.mean(axis=1)

    def save(self, path):
        joblib.dump(self.models, path)

    def load(self, path):
        self.models = joblib.load(path)