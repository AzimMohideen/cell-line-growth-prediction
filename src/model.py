from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

def apply_pca(X_train, X_test, components=10):
    pca = PCA(n_components=components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
