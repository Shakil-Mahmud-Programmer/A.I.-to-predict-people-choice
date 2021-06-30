import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("choice.csv")
x=df.drop(columns=["choice"])
y=df["choice"]
model=DecisionTreeClassifier()
model.fit(x,y)
age=int(input("Age of the person:"))
gender=int(input("gender of the person(Male=1 & Female=0):"))
predict=model.predict([[age,gender]])
print(predict)
Plot=plt.scatter(df["age"],df["choice"],color='red')
plt.title("Choice",color='purple')
plt.legend(["Choice"])
plt.show()