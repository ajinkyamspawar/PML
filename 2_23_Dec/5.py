import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import os 

os.chdir(r"C:\LAB PROGRAMS\9_MACHINE LEARNING\Cases\Bankruptcy")
bankruptcy=pd.read_csv("Bankruptcy.csv")
X=bankruptcy.drop(['NO','D','YR'],axis=1)
y=bankruptcy['D']


le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

X_train,X_test,y_train,y_test=train_test_split(X,le_y,stratify=le_y,random_state=2022,train_size=0.7)
scaler= StandardScaler()
knn=KNeighborsClassifier(n_neighbors=3)

pipe=Pipeline([('STD',scaler),('KNN',knn)])
pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
print(accuracy_score(y_test,y_pred))

y_pred_prob=pipe.predict_proba(X_test)
print(log_loss(y_test,y_pred))

########## Grid Search ############


from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
scaler= StandardScaler()
knn=KNeighborsClassifier()
pipe=Pipeline([('STD',scaler),('KNN',knn)])
kfold= KFold(n_splits=5,shuffle= True, random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16,2)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,verbose=3,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


#########Predicting on unlabelled data

tst_bankruptcy =pd.read_csv('testBankruptcy.csv',index_col=0)
best_model= gcv.best_estimator_
predictions=best_model.predict(tst_bankruptcy)
print(tst_bankruptcy)
















