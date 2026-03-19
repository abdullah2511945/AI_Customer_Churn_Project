from sklearn.linear_model import LogisticRegression

def train_baseline_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def train_optimized_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        C=0.5,          
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    return model