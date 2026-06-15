import joblib
import pandas as pd

def inspect_model():
    try:
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        expected_columns = joblib.load('expected_columns.pkl')
        
        print("--- Model Information ---")
        print(f"Model Type: {type(model).__name__}")
        print(f"Parameters: {model.get_params()}")
        
        if hasattr(model, 'feature_importances_'):
            print("\nFeature Importances:")
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'feature': expected_columns, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            print(feature_importance_df.head(10))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_model()
