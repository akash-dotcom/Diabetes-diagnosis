import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

df=pd.read_csv(r'C:\Users\AKASH\OneDrive\Desktop\Conatus  Project\data\diabetes.csv')
# print(df.head(5))

X = df.iloc[:,:-1].values
y = df.Outcome
# print(Y)

X_train ,X_test , y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=10)
# print('Train Set:',X_train.shape,y_train.shape)
# print('Test Set:',X_test.shape,y_test.shape)



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
predictions = logreg.predict(X_test)
print(predictions)
print(X_test)

from sklearn.metrics import confusion_matrix
i=confusion_matrix(y_test,predictions)
print(i)


from sklearn.metrics import accuracy_score
j=accuracy_score(y_test,predictions)
print(j)


if not os.path.exists('models'):
    os.makedirs('models')

MODEL_PATH = "models/logistic_reg.sav"
pickle.dump(logreg,open(MODEL_PATH,'wb'))

data=[[5,166,72,19,27,25.8,0.587,51]]
df=pd.DataFrame(data,columns=['pregnant','insulin','Skin Thickness','bmi','age','glucose','bp','pedigree'])

new_pred = logreg.predict(df)
print(new_pred)












