import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os


concrete= pd.read_csv("Concrete_Data.csv")


X=concrete.drop('Strength',axis=1)
y=concrete['Strength']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2022,train_size=0.7)


elasticnet=ElasticNet()
elasticnet.fit(X_train,y_train)
y_pred=elasticnet.predict(X_test)

print(r2_score(y_test,y_pred))


#Grid 

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

elasticnet=ElasticNet()

params={'alpha':np.linspace(0.001,11,20),'l1_ratio':np.linspace(0,1,5)}
gcv=GridSearchCV(elasticnet,param_grid=params,cv=kfold,scoring='r2')

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model=gcv.best_estimator_
print(best_model.coef)



























