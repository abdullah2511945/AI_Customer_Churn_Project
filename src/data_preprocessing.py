import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # 1️⃣ Clean column names
    df.columns = df.columns.str.strip()

    # 2️⃣ Drop customerID safely (if present)
    df = df.drop("customerID", axis=1, errors="ignore")

    # 3️⃣ Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 4️⃣ Drop rows with missing critical values
    required_cols = ["TotalCharges", "Churn"]
    existing_required = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=existing_required)

        # Ensure Churn exists
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found")

        # Clean and encode target
    df["Churn"] = df["Churn"].astype(str).str.strip()

    y = df["Churn"].replace({"Yes": 1, "No": 0})

    # Force numeric
    y = pd.to_numeric(y, errors="raise")

    # Final validation
    assert set(y.unique()).issubset({0, 1}), f"Unexpected labels: {y.unique()}"

    # 6️⃣ Encode target (Yes/No → 1/0)
    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})

    # Safety check
    if y.isnull().sum() > 0:
        raise ValueError("Target variable contains invalid values after mapping")

        # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 🚨 Step 1: Check cardinality
    for col in cat_cols:
        print(f"{col}: {X[col].nunique()} unique values")

    # 🚨 Step 2: Drop high-cardinality columns
    high_card_cols = [col for col in cat_cols if X[col].nunique() > 5]

    print("Dropping high-cardinality columns:", high_card_cols)

    X = X.drop(columns=high_card_cols)

    # Step 3: Encode remaining columns
    X = pd.get_dummies(X, drop_first=True)

    # 8️⃣ Final safety checks
    if X.isnull().sum().sum() > 0:
        raise ValueError("NaN values detected in features after preprocessing")

    # 9️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test