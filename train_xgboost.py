import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_meta_model():
    print("🚀 Loading training data...")
    # 1. Load the dataset
    df = pd.read_csv("dummy_training_data.csv")
    
    # 2. Define our Features (X) and Target (y)
    # These MUST match the exact order of FEATURE_NAMES in shap_explainer.py
    feature_columns = [
        "zone_count", 
        "door_count", 
        "window_count", 
        "zones_missing_door", 
        "zones_missing_window",
        "gpt_compliance_score",  # Stage 2 Input
        "gpt_status_encoded"     # Stage 2 Input
    ]
    
    X = df[feature_columns]
    y = df["Final_Approved_Score"] # We are predicting the final human score
    
    # 3. Split data: 80% for training, 20% for testing the model's accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 Training XGBoost Regressor...")
    # 4. Initialize and train the model
    # We use a Regressor because we want a score from 0-100 (continuous number)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,      # Number of decision trees
        max_depth=4,           # How deep the trees can think
        learning_rate=0.05,    # How fast it learns
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate the model on the 20% it hasn't seen
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"📊 Model Accuracy Check: The model's predictions are off by an average of {mae:.2f} points.")
    
    # 6. Save the trained model for the Streamlit app
    os.makedirs("models", exist_ok=True)
    model_path = "models/feasibility_xgb.ubj"
    model.save_model(model_path)
    print(f"✅ Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_meta_model()