import pandas as pd
import joblib
import os
import lightgbm as lgb

class NTargetModel:
    def __init__(self, target_names, model_params=None):
        self.target_names = target_names
        self.models = {}
        
        # Parametri ottimali da esperimenti: LightGBM (leaves=50)
        if model_params is None:
            self.model_params = {
                "n_estimators": 150,
                "num_leaves": 50,
                "learning_rate": 0.05,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }
        else:
            self.model_params = model_params

    def train(self, X, y):
        """Addestra un modello LightGBM per ogni target."""
        for target in self.target_names:
            if target in y.columns:
                print(f"Training LightGBM for {target}...")
                
                # Inizializza il regressore LightGBM
                model = lgb.LGBMRegressor(**self.model_params)
                
                valid_mask = y[target].notna()
                if valid_mask.sum() > 0:
                    model.fit(X[valid_mask], y.loc[valid_mask, target])
                    self.models[target] = model
                else:
                    print(f"Warning: No valid data for {target}")
            else:
                print(f"Warning: {target} not found.")

    def predict(self, X):
        """Fa predizioni per tutti i target e restituisce la media."""
        if not self.models:
            raise ValueError("Model not trained yet!")
            
        predictions = pd.DataFrame(index=X.index)
        
        for target, model in self.models.items():
            predictions[target] = model.predict(X)
            
        if predictions.empty:
            return pd.Series(0, index=X.index)
            
        return predictions.mean(axis=1)

    def save(self, directory):
        path = os.path.join(directory, "n_target_model.joblib")
        joblib.dump(self.models, path)

    def load(self, directory):
        path = os.path.join(directory, "n_target_model.joblib")
        self.models = joblib.load(path)