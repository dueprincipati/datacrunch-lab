import pandas as pd
from scipy.stats import spearmanr
from src.main import train, infer

def run_experiment():
    print("--- Caricamento Dati Locali ---")
    try:
        # Assumiamo che i dati siano stati scaricati in 'data/'
        X_train = pd.read_parquet("data/X_train.parquet")
        y_train = pd.read_parquet("data/y_train.parquet")
        X_test = pd.read_parquet("data/X_test_reduced.parquet") # Usato come validation set qui
        # Per validare localmente ci serve y_test, che solitamente è una parte splittata di y_train
        # Simulo uno split temporale per il test
    except Exception as e:
        print(f"Errore caricamento dati: {e}")
        return

    # Split Train/Val (Simulazione Walk-Forward semplificata)
    moons = X_train["moon"].unique()
    split_point = int(len(moons) * 0.8)
    train_moons = moons[:split_point]
    val_moons = moons[split_point:]

    print(f"Training su {len(train_moons)} moons, Validazione su {len(val_moons)} moons")

    X_t = X_train[X_train["moon"].isin(train_moons)]
    y_t = y_train[y_train["moon"].isin(train_moons)]
    X_v = X_train[X_train["moon"].isin(val_moons)]
    y_v = y_train[y_train["moon"].isin(val_moons)]

    # 1. Esegui Training
    train(X_t, y_t, "resources_test")

    # 2. Esegui Inferenza
    preds = infer(X_v, "resources_test")

    # 3. Calcola Metrica (Spearman su target_b o target principale)
    # Uniamo le predizioni con i valori veri
    score_df = pd.merge(preds, y_v, on=["id", "date" if "date" in preds.columns else "moon"])
    
    # Calcolo correlazione media per moon
    correlations = []
    for moon, group in score_df.groupby("date" if "date" in score_df.columns else "moon"):
        corr = spearmanr(group["prediction"], group["target_b"]).correlation
        correlations.append(corr)
    
    avg_corr = sum(correlations) / len(correlations)
    print(f"\n=== RISULTATO SPERIMENTALE ===")
    print(f"Mean Spearman Correlation: {avg_corr:.4f}")

if __name__ == "__main__":
    run_experiment()