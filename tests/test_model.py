
from src.data_preprocessing import load_data, preprocess, split_features
from sklearn.ensemble import RandomForestClassifier


def test_simple_model_runs():
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_features(df)
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    X_train2 = X_train[numeric_cols]
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X_train2, y_train)
    preds = clf.predict(X_train2)
    assert len(preds) == X_train2.shape[0]
