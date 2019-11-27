from numpy import sqrt
from pandas import DataFrame
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def LinearRegression_model(xtrain , xtest , ytrain , ytest):

    model = LinearRegression().fit(xtrain , ytrain)
    log_detial = {'Model' : '' , 'Train-S' : 0 , 'Test-S' : 0 , 'R2' : 0 , 'RMSE' : 0 , 'AE' : 0}
    log_detial['Model'] = 'LinearRegression'
    log_detial['Train-S'] = model.score(xtrain,ytrain)
    log_detial['Test-S'] = model.score(xtest,ytest)
    log_detial['R2'] = r2_score(ytest,model.predict(xtest))
    log_detial['RMSE'] = sqrt(mean_squared_error(ytest,model.predict(xtest)))
    log_detial['AE'] = mean_absolute_error(ytest,model.predict(xtest))
    return log_detial

def Lasso_model(xtrain, xtest, ytrain, ytest):

    model = Lasso().fit(xtrain, ytrain)
    log_detial = {'Model': '', 'Train-S': 0, 'Test-S': 0, 'R2': 0, 'RMSE': 0, 'AE': 0}
    log_detial['Model'] = 'LassoRegression'
    log_detial['Train-S'] = model.score(xtrain, ytrain)
    log_detial['Test-S'] = model.score(xtest, ytest)
    log_detial['R2'] = r2_score(ytest, model.predict(xtest))
    log_detial['RMSE'] = sqrt(mean_squared_error(ytest, model.predict(xtest)))
    log_detial['AE'] = mean_absolute_error(ytest, model.predict(xtest))
    return log_detial


def Rigde_model(xtrain, xtest, ytrain, ytest):

    model = Ridge().fit(xtrain , ytrain)
    log_detial = {'Model': '', 'Train-S': 0, 'Test-S': 0, 'R2': 0, 'RMSE': 0, 'AE': 0}
    log_detial['Model'] = 'RigdeRegression'
    log_detial['Train-S'] = model.score(xtrain, ytrain)
    log_detial['Test-S'] = model.score(xtest, ytest)
    log_detial['R2'] = r2_score(ytest, model.predict(xtest))
    log_detial['RMSE'] = sqrt(mean_squared_error(ytest, model.predict(xtest)))
    log_detial['AE'] = mean_absolute_error(ytest, model.predict(xtest))
    return log_detial


def DecisionTreeRegressor_model(xtrain, xtest, ytrain, ytest):

    model = DecisionTreeRegressor().fit(xtrain , ytrain)
    log_detial = {'Model': '', 'Train-S': 0, 'Test-S': 0, 'R2': 0, 'RMSE': 0, 'AE': 0}
    log_detial['Model'] = 'DecisionTreeRegressor'
    log_detial['Train-S'] = model.score(xtrain, ytrain)
    log_detial['Test-S'] = model.score(xtest, ytest)
    log_detial['R2'] = r2_score(ytest, model.predict(xtest))
    log_detial['RMSE'] = sqrt(mean_squared_error(ytest, model.predict(xtest)))
    log_detial['AE'] = mean_absolute_error(ytest, model.predict(xtest))
    return log_detial

base_model = [
    LinearRegression_model,
    Lasso_model,
    Rigde_model,
    DecisionTreeRegressor_model
]

def Feature_Selection_Select(k , function , model , xtrain, xtest, ytrain, ytest  ):

    clf = Pipeline([
        ('feature_selection', SelectFromModel(k)),
        ('regression', model)
    ])

    clf.fit(xtrain, ytrain)
    log_detial = {'Model': '', 'Select-Method' : '' , 'Select-Function' : '' , 'Feature-Count': 0, 'Train-S': 0, 'Test-S': 0, 'R2': 0, 'RMSE': 0, 'AE': 0}
    log_detial['Model'] = str(model.__class__).split('.')[-1].replace("'>" , '')
    log_detial['Select-Method'] = "SelectFromModel"
    log_detial['Select-Function'] = function
    log_detial['Feature-Count'] = clf.named_steps['feature_selection'].get_support().sum()
    log_detial['Train-S'] = clf.score(xtrain, ytrain)
    log_detial['Test-S'] = clf.score(xtest, ytest)
    log_detial['R2'] = r2_score(ytest, clf.predict(xtest))
    log_detial['RMSE'] = sqrt(mean_squared_error(ytest, clf.predict(xtest)))
    log_detial['AE'] = mean_absolute_error(ytest, clf.predict(xtest))
    return (log_detial)

def Feature_Selection_Recursive(k , function , model , xtrain, xtest, ytrain, ytest  ):

    selector = RFE(function, k)
    selector = selector.fit(xtrain, ytrain)

    xtrain = selector.transform(xtrain)
    xtest = selector.transform(xtest)
    clf = model

    clf.fit(xtrain, ytrain)
    log_detial = {'Model': '', 'Select-Method' : '' , 'Select-Function' : '' , 'Feature-Count': 0, 'Train-S': 0, 'Test-S': 0, 'R2': 0, 'RMSE': 0, 'AE': 0}
    log_detial['Model'] = str(clf.__class__).split('.')[-1].replace("'>" , '')
    log_detial['Select-Method'] = 'Recursive'
    log_detial['Select-Function'] = str(function.__class__).split('.')[-1].replace("'>" , '')
    log_detial['Feature-Count'] = k
    log_detial['Train-S'] = clf.score(xtrain, ytrain)
    log_detial['Test-S'] = clf.score(xtest, ytest)
    log_detial['R2'] = r2_score(ytest, clf.predict(xtest))
    log_detial['RMSE'] = sqrt(mean_squared_error(ytest, clf.predict(xtest)))
    log_detial['AE'] = mean_absolute_error(ytest, clf.predict(xtest))
    return (log_detial)

def Base_Model_Profile(xtrain, xtest, ytrain, ytest):
    log = list()
    for i in base_model:
        log.append(i(xtrain, xtest, ytrain, ytest))
    return DataFrame(log)

def Feature_Selection_Profile(xtrain, xtest, ytrain, ytest):
    log = list()

    for i in [Ridge(), Lasso(), DecisionTreeRegressor()]:
        for j in [LinearRegression() , DecisionTreeRegressor()]:
            log.append(
                Feature_Selection_Select(i,str(i.__class__).split('.')[-1].replace("'>", ''),j,xtrain, xtest, ytrain, ytest)
            )

    for i in range(1 , xtrain.shape[1]):
        for j in [LinearRegression() , DecisionTreeRegressor()]:
            for k in [LinearRegression() , DecisionTreeRegressor()]:
                log.append(
                    Feature_Selection_Recursive(i, j , k , xtrain, xtest, ytrain, ytest)
                )

    return DataFrame(log)

def test_data():
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.30, random_state=0)
    return  xtrain, xtest, ytrain, ytest

def test_base_profile():
    xtrain, xtest, ytrain, ytest = test_data()
    print(Base_Model_Profile(xtrain, xtest, ytrain, ytest))

def test_Feature_Selection_Profile():
    xtrain, xtest, ytrain, ytest = test_data()
    print(Feature_Selection_Profile(xtrain, xtest, ytrain, ytest))


# test_base_profile()
# test_Feature_Selection_Profile()