import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 

job= pd.read_csv("JobSalary2.csv")

## Finding NAs in columns 
job.isnull().sum()

#dropping the rows with NA values 
job.dropna()


#Constant Imputer
imp= SimpleImputer(strategy='constant',fill_value=50)
imp.fit_transform(job)

job.mean()

# Mean Imputer
imp=SimpleImputer(strategy='mean')
imp.fit_transform(job)

# Median Imputer
imp=SimpleImputer(strategy='median')
np_imp=imp.fit_transform(job)

pd_imp=pd.DataFrame(np_imp,columns=job.columns)
print(pd_imp)

#############chemical Process###############
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

os.chdir(r"C:\LAB PROGRAMS\9_MACHINE LEARNING\Cases\Chemical Process Data")

chemdata=pd.read_csv("ChemicalProcess.csv")

##Finding Nas in columns 

chemdata.isnull().sum()
X=chemdata.drop("Yield",axis=1)
y=chemdata['Yield']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=202,train_size=0.7)

############Linear regression#############

imp=SimpleImputer(strategy='mean')
X_trn_trf=imp.fit_transform(X_train)
X_tst_trf=imp.transform(X_test)


lr=LinearRegression()

lr.fit(X_trn_trf,y_train)

y_pred =lr.predict(X_tst_trf)

print(r2_score(y_test,y_pred))

##### With Pipeline ##########

from sklearn.pipeline import Pipeline

imp=SimpleImputer(strategy='median')
lr=LinearRegression()
pipe=Pipeline([('IMPUTE',imp),('LR',lr)])
pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print(r2_score(y_test,y_pred))

########### K-NN ############
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.impute import SimpleImputer 


#MEDIAN

imp=SimpleImputer(strategy='median')
scaler= StandardScaler()
knn=KNeighborsRegressor()
pipe= Pipeline([('IMPUTE',imp),('STD',scaler),('KNN',knn)]) 
kfold= KFold(n_splits=5,shuffle= True, random_state=2022)
params={'KNN__n_neighbors':np.arange(1,11)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#MEAN

imp=SimpleImputer(strategy='mean')
scaler= StandardScaler()
knn=KNeighborsRegressor()
pipe= Pipeline([('IMPUTE',imp),('STD',scaler),('KNN',knn)]) 
kfold= KFold(n_splits=5,shuffle= True, random_state=2022)
params={'KNN__n_neighbors':np.arange(1,11)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


#COMBINE

imp=SimpleImputer()
scaler= StandardScaler()
knn=KNeighborsRegressor()
pipe= Pipeline([('IMPUTE',imp),('STD',scaler),('KNN',knn)]) 
print(pipe.get_params())
kfold= KFold(n_splits=5,shuffle= True, random_state=2022)
params={'IMPUTE__strategy':['mean','median'],
        'KNN__n_neighbors':np.arange(1,11)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)







