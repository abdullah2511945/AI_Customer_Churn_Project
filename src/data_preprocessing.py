import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df

import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Drop customer ID if exists
    df = df.drop("customerID", axis=1, errors="ignore")
    
    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop rows with missing required columns
    required_cols = ["TotalCharges", "Churn"]
    existing_required = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=existing_required)
    
    # Clean target column
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found")
    df["Churn"] = df["Churn"].astype(str).str.strip()
    
    # Map target to numeric
    y = df["Churn"].replace({"Yes": 1, "No": 0})
    if y.isnull().sum() > 0:
        raise ValueError(f"Target contains invalid values at indices: {y[y.isnull()].index}")
    
    # Features
    X = df.drop(columns=["Churn"])
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    # Drop high-cardinality columns
    high_card_cols = [col for col in cat_cols if X[col].nunique() > 5]
    print("Dropping high-cardinality columns:", high_card_cols)
    X = X.drop(columns=high_card_cols)
    
    # One-hot encode remaining categorical columns
    X = pd.get_dummies(X, drop_first=True)
    
    # Final NaN check
    if X.isnull().sum().sum() > 0:
        raise ValueError("NaN values detected in features after preprocessing")
    
    # ✅ Now X and y have same length
    print(f"Final shapes -> X: {X.shape}, y: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test