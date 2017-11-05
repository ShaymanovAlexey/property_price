import numpy
import numpy as np
import pandas
from matplotlib import pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

import copy

from sklearn.model_selection import GridSearchCV   #Perforing grid search

train = pandas.read_csv('Train.csv')
test = pandas.read_csv('Test.csv')
copy_test =copy.copy(test)
train = train.replace(numpy.nan, -999)

train['g_lift'] = train['g_lift'].replace(-999, 0)

test = test.replace(numpy.nan, -999)

COLUMNS = ['street_id', 'build_tech', 'floor', 'area', 'rooms', 'balcon', 'metro_dist', 'g_lift', 'n_photos', 'kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'kw6', 'kw7', 'kw8', 'kw9', 'kw10', 'kw11', 'kw12', 'kw13']

y = train['price'].values
X = train[COLUMNS].values
Xt = test[COLUMNS].values

train['new_room'] = train['area']/train['rooms']

train['stage'] = train['area']*train['n_photos']

test['new_room'] = test['area']/test['rooms']

test['stage'] = test['area']*test['n_photos']

test_head = [x for x in test.columns if x not in ['id','date']]
test = test[test_head]


from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn import model_selection

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['price'])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    kfold = model_selection.KFold(n_splits=10, random_state=10)
    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain['price'], cv=kfold, scoring= 'neg_mean_squared_error')

    # Print model report:
    print("\nModel Report")
    #print("Accuracy : %.4g" % mean_absolute_error(dtrain['price'].values, dtrain_predictions))
    print("Accuracy Square: %.4g" % mean_squared_error(dtrain['price'].values, dtrain_predictions))
    #print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        plt.ion()
        plt.draw()
        feat_imp = pandas.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.ioff()
        plt.show()



#print(pandas.DataFrame(X))

target = 'price'

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def somefunction():
    mdl = GradientBoostingRegressor(random_state=0)

    predictors = [x for x in train.columns if x not in ["price",'date','id']]

    #param_test1 = {'n_estimators':list(range(1080,1101,10))}
    #param_test2 = {'max_depth': list(range(5, 16, 2)), 'min_samples_split': list(range(200, 1001, 200))}
    #param_test3 = {'min_samples_leaf': list(range(30, 71, 10))}
    #param_test4 = {'max_features': list(range(7, 23, 1))}
    param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    #gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(max_features=20, n_estimators = 1090, min_samples_split=600,max_depth=5,learning_rate=0.8,min_samples_leaf= 70, subsample=0.9,random_state=10), param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
    #gsearch1.fit(train[predictors],train[target])

    mdl1_tuned = GradientBoostingRegressor(learning_rate=0.2, n_estimators=4360,min_samples_leaf=70,min_samples_split=600, max_depth=5,max_features=20,subsample=0.9,random_state=10)


    #print(gsearch1.best_params_, gsearch1.best_score_)
    modelfit(mdl1_tuned, train, predictors)
    preds = mdl1_tuned.predict(test)
    copy_test['price'] = preds
    copy_test[['id', 'price']].to_csv('sub.csv', index=False)
    #mdl2_tuned = GradientBoostingRegressor(max_features=18, learning_rate=0.01, n_estimators=2420, min_samples_split=200, max_depth=11, subsample=0.9, random_state=10)
    #modelfit(mdl2_tuned, train, predictors)

if __name__ == '__main__':
    somefunction()
#print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
#modelfit(mdl, train, predictors)

#mdl.fit(X, y)

#X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2)

#model = mdl.fit(X_train, y_train)

#predictions = mdl.predict(X_test)

#print("Score",mdl.score(X_test, y_test))
'''
plt.ion()
plt.draw()
plt.scatter(y_test, predictions)
plt.xlabel("True_Values")
plt.ylabel("Predictions")

plt.ioff()
plt.show()
'''

"""
mdl = RandomForestRegressor(oob_score=True)

mdl2 = GradientBoostingRegressor(max_depth=20)

mdl.fit(X, y)

mdl2.fit(X,y)

preds = mdl.predict(Xt)

preds2 = mdl2.predict(Xt)

test['price'] = preds2

test[['id', 'price']].to_csv('sub.csv', index=False)

y_nump = copy.deepcopy(y)


y = pandas.DataFrame(y)

print(y_nump)

pred = pandas.DataFrame(mdl.oob_prediction_)

l1 = mdl.oob_prediction_

#l2 = mdl2.oob_improvement_

#r2_score_data = mean_absolute_error(y_nump,l1)

#r2_score_data2 = mean_absolute_error(y_nump,l2)


#print("1", r2_score_data)

#print("2", r2_score_data2)

#print ("AUC-ROC (oob) = ", accuracy_score(y, mdl.oob_prediction_))

#print ("AUC-ROC (oob) = ", roc_auc_score(y, mdl.oob_prediction_))

"""
