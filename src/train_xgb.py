from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    

    # 🔒 Force encoding if something slipped through
    if y_train.dtype == 'object':
        y_train = y_train.replace({"Yes": 1, "No": 0})
        y_train = y_train.astype(int)

    model = XGBClassifier(
        n_estimators=30,        
        max_depth=3,            
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1      
    )

    model.fit(X_train, y_train)
    return model