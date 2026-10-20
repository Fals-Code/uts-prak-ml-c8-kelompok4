"""Implementasi Decision Tree sebagai model pembanding."""

from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train, random_state: int = 42) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_decision_tree(model: DecisionTreeClassifier, X_test):
    return model.predict(X_test)
