from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

"""

"""

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

# using this only for data_binary.csv
def train_logistic_binary_regression(X_train, y_train):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_logistic_multi_regression(X_train, y_train, scale_data=True, max_iter=1000, solver='lbfgs'):
    if scale_data:
        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train)
    else:
        X_train_transformed = X_train
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(multi_class='multinomial', solver=solver, max_iter=max_iter)
    model.fit(X_train_transformed, y_train)
    
    return model, X_train_transformed


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    # return 
    return model

# need to add bagging classifier

def train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42):
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model
