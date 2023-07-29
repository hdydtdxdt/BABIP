from sklearn.metrics import accuracy_score


def is_overfit(model, X_train, y_train, X_test, y_test):
    pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, pred_train)
    
    pred_test = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, pred_test)
    
    accuracy_interval = abs(accuracy_train - accuracy_test)

    if accuracy_interval >= 0.1:
        return True
    else:
        return False
