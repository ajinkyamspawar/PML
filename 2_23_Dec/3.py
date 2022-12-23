import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report



############# Medical Cost Insurance ###############
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


os.chdir(r"C:\LAB PROGRAMS\9_MACHINE LEARNING\Cases\Medical Cost Personal")
insurance=pd.read_csv("insurance.csv")
dum_ins =pd.get_dummies(insurance,drop_first=True)

x= dum_ins.drop('charges',axis=1)
y=dum_ins['charges']

kfold= KFold(n_splits=5,shuffle= True, random_state=2022)
lr=LinearRegression()
result= cross_val_score(lr,x,y,cv=kfold,scoring='r2')
print(result.mean())

##k-NN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
X= dum_ins.drop('charges',axis=1)
y=dum_ins['charges']
scaler=StandardScaler()
knn=KNeighborsRegressor()
pipe=Pipeline([('STD',scaler),('KNN',knn)])
kfold= KFold(n_splits=5,shuffle= True, random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


###Predicting on labelled data

knn=KNeighborsRegressor(n_neighbors=7)
pipe=Pipeline([('STD',scaler),('KNN',knn)])
pipe.fit(X,y)


tst_insure= pd.read_csv("tst_insure.csv")

dum_tst=pd.get_dummies(tst_insure,drop_first=True)

print(X.dtypes)
print(dum_tst.dtypes)
predictions=pipe.predict(dum_tst)


# or using Grid Search
 
pd_cv =pd.DataFrame(gcv.cv_results_)

best_model= gcv.best_estimator_

tst_insure =pd.read_csv("tst_insure.csv")
dum_tst=pd.get_dummies(tst_insure,drop_first=True)

predictions= best_model.predict(dum_tst)

predictions




























