import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns 

#import plotly.express as px

wine_data = pd.read_csv("/Users/vipul1/Downloads/winequality-white.csv",sep=';')
wine_data.head()

corr = wine_data.corr()
plt.pyplot.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

wine_data['goodquality'] = [1 if x >= 7 else 0 for x in wine_data['quality']]

X=wine_data.drop(['quality','goodquality'],axis=1)
y=wine_data['goodquality']
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
#standardisation
X= StandardScaler().fit_transform(X)
#splitting
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=.2,random_state=0)
#model_fitting
model=DecisionTreeClassifier(random_state=1)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
#accuracy_calc
print(accuracy_score(y_test,predictions))
print(mean_absolute_error(y_test,predictions))

def mae(max_leaf,X_train,X_test,y_train,y_test):
    model=DecisionTreeClassifier(max_leaf_nodes=max_leaf,random_state=1)
    model.fit(X_train,y_train)
    preds=model.predict(X_test)
    mae=mean_absolute_error(y_test,preds)
    return(mae)
score = {leaf_size: mae(leaf_size, X_train, X_test, y_train, y_test) for leaf_size in range(2,1500)}
best_tree_size = min(score, key=score.get)
print(best_tree_size)
final_model=DecisionTreeClassifier(max_leaf_nodes=best_tree_size,random_state=1)
final_model.fit(X_train,y_train)
preds_final=final_model.predict(X_test)
mean_absolute_error(y_test,preds_final)

model1=RandomForestRegressor(random_state=1)
model1.fit(X_train,y_train)
pred=model1.predict(X_test)
mean_absolute_error(y_test,pred)
