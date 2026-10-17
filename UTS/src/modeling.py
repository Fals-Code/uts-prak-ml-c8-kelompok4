from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train, random_state=42):
    """Melatih model Decision Tree menggunakan data training."""
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_decision_tree(model, X_test):
    """Melakukan prediksi menggunakan model Decision Tree."""
    return model.predict(X_test)
