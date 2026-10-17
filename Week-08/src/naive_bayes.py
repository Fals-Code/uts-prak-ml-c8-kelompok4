"""Implementasi Gaussian Naive Bayes."""

from sklearn.naive_bayes import GaussianNB


def train_naive_bayes(X_train, y_train) -> GaussianNB:
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict_naive_bayes(model: GaussianNB, X_test):
    return model.predict(X_test)


