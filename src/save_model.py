import joblib

def save_model(model, filename="model.pkl"):
    joblib.dump(model, filename)