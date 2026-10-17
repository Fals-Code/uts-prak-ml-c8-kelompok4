"""Implementasi Support Vector Machine dengan konfigurasi dasar."""

from sklearn.svm import SVC


def train_svm(X_train, y_train, random_state: int = 42) -> SVC:
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_svm(model: SVC, X_test):
    return model.predict(X_test)


