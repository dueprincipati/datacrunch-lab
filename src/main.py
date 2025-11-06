import pandas as pd
import joblib 
from sklearn.linear_model import LinearRegression

# --- Costants ---
# Define the target we want to focus on
# target_b has got the higher payout
TARGET_NAME = "target_b"
# column's name for prediction
PREDICTION_NAME = "prediction"

def get_feature_names(X_data: pd.DataFrame) -> list:
    """Get the feature whose names start with 'gordon_' and 'dolly_'.
    documentation gathers them in families.
    """
    features = [col for col in X_data.columns if col.startswith("gordon_") or col.startswith("dolly_")]
    return features

def train (Xtrain: pd.DataFrame, y_train: pd.DataFrame, model_directory_path: str):
    """
    Train a baseline model and save it to disk.
    based on the quickstarter architecture.
    """

    print("Starting training baseline model...")

    # 1. feature selection
    feature_names = get_feature_names(Xtrain)

    # 2. to unify X and y for data matching
    train_data = pd.merge(X_train, y_train, on=["id", "moon"])

    # 3. to manage NAN values for linear regression
    train_data[feature_names] = train_data[feature_names].fillna(0)

    # 4. To extract only the target we want to predict
    train_data=train_data.dropna(subset=TARGET_NAME)
    y_target = train_data [TARGET_NAME]

    # 5. model training
    model = LinearRegression() [2]
    model.fit(train_data[feature_names], y_target)

    # 6. save the model to disk
    os.makedirs(model_directory_path, exist_ok=True)
    model_path = os.path.join(model_directory_path, "baseline_model.joblib")
    joblib.dump(model, model_path)

    print(f"Model training completed. Model saved to {model_path}")

def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    """
    Load the trained model from disk and make predictions on the test set 
"""
    print("Starting baseline inference...")

    # 1. load the model from disk
    model_path = os.path.join(model_directory_path, "baseline_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}; please train the model first.")
    
    model = joblib.load(model_path)

    # 2. feature selection
    feature_names = get_feature_names(X_test)

    # 3. to manage NAN values for linear regression
    X_test_processed = X_test[feature_names].fillna(0)

    # 4. make predictions
    predictions = model.predict(X_test_processed)

    # 5. prepare the output DataFrame
    inference_df  = pd.DataFrame({
        "id": X_test["id"],
        "moon": X_test["moon"],
        PREDICTION_NAME: predictions
    })

    print("Inference completed.")
    return inference_df

# --- local test ---
# this block will be executed only when you execute 'python src/main.py'
if __name__ == "__main__":
    print("Running local test...")

    # 1. Load sample data
    try:
        X_train_full = pd.read_parquet("data/X_train.parquet")
        y_train_full = pd.read_parquet("data/y_train.parquet")
    except FileNotFoundError:
        print("Sample data not found. Please ensure 'data/X_train.parquet' and 'data/y_train.parquet' exist.")
        exit()

    # We use only the first 2 moons for quick local testing
    # moon are sequential integers
    all_moons = X_train_full["moon"].unique()
    moons_to_train = all_moons[:2] # first two moons for training
    moon_to_infer = all_moons[2] # third moon for inference

    X_train_sample = X_train_full[X_train_full["moon"].isin(moons_to_train)]
    y_train_sample = y_train_full[y_train_full["moon"].isin(moons_to_train)]
    X_test_sample = X_train_full[X_train_full["moon"] == moon_to_infer]

    print(f"Training on moons: {moons_to_train}")
    print(f"Inferring on moon: {moon_to_infer}")

    # 2. define model directory path for this local test
    TEST_MODEL_DIR = "models/test_baseline_model/"

    # 3. Train the model
    train(X_train_sample, y_train_sample, TEST_MODEL_DIR)

    # 4. Run inference
    predictions_df = infer(X_test_sample, TEST_MODEL_DIR)

    print("\n--- Inference Results (first 5 rows) ---")
    print(predictions_df.head())
    print(f"\nDataFrame shape: {predictions_df.shape}")
    print("\nLocal test successfully completed!") 
          