
"""Train a RandomForest classifier and save the model."""
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.data_preprocessing import load_data, preprocess, split_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_evaluate():
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_features(df)
    # simple numeric-only model
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    X_train2 = X_train[numeric_cols]
    X_test2 = X_test[numeric_cols]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train2, y_train)
    y_pred = clf.predict(X_test2)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    print(classification_report(y_test, y_pred))
    dump(clf, os.path.join(MODEL_DIR, 'rf_tsunami.joblib'))
    print('Model saved to', MODEL_DIR)

if __name__ == '__main__':
    train_and_evaluate()
