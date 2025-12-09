import pandas as pd

def get_feature_names(df: pd.DataFrame) -> list:
    """Seleziona le colonne gordon_ e dolly_."""
    return [c for c in df.columns if c.startswith("gordon_") or c.startswith("dolly_")]

def preprocess_features(df: pd.DataFrame, feature_names: list = None) -> pd.DataFrame:
    """Gestione base dei NaN e selezione."""
    if feature_names is None:
        feature_names = get_feature_names(df)
    
    # Riempimento NaN semplice con 0 (puoi cambiarlo con media/mediana qui)
    return df[feature_names].fillna(0)