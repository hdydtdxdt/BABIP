from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search(model, parameters, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid=parameters, scoring='accuracy', n_jobs=-1, refit=True, cv=5, verbose=1, return_train_score=True)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_

def random_search(model, parameters, X_train, y_train):
    randomized_search = RandomizedSearchCV(model, param_distributions=parameters, n_iter=10, return_train_score=True, random_state=555)
    randomized_search.fit(X_train, y_train)

    return randomized_search.best_params_
