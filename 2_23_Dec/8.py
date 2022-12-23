import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

concrete= pd.read_csv("Concrete_Data.csv")


X=concrete.drop('Strength',axis=1)
y=concrete['Strength']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2022,train_size=0.7)


ridge=Ridge()
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)

print(r2_score(y_test,y_pred))


#Grid 

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)

ridge=Ridge()

params={'alpha':np.linspace(0.001,11,20)}
gcv=GridSearchCV(ridge,param_grid=params,cv=kfold,scoring='r2')

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
