import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mlt
#importing data set
data=pd.read_csv('diamonds.csv')
data.columns
data.isna().sum()
########################3
data=pd.read_csv('diamonds.csv',na_values=[0])

columns=['carat','colour','cut','clarity','depth','table','price']

for column in columns:
    mean=data[column].mean()
    data[column]=data[column].replace(0,mean)
    
    ######################################
#data['carat'].mean()
#data[column]=data[column].replace(0,mean)
#######################################
plt.figure(figsize=(12,4))
sns.heatmap(data.corr(),annot=True)
###################################
from sklearn.model_selection import train_test_spit
x_train,x_test,y_train,y_test=train_test_split(x,y,
                            test_size=0.20,
                            random_state=0)
#####################
from sklearn.linear_model import LogisticRegression()
classifier.fit(x_train,y_train)
########################
y_pred=classifier.predict(x_test)
###################
from sklearn import metrics

matrics.confusion_matrix(y_test,y_pred)
metrics.accuracy_score(y_test,y_pred)

print(metrics.classication_repot(y_test,y_pred))