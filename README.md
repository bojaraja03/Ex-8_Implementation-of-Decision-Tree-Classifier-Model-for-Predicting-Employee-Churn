# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: BOJA RAJA G
RegisterNumber:  212225230036
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['stayed','left'],filled=True)
plt.show()
```

## Output:
<img width="1248" height="242" alt="image" src="https://github.com/user-attachments/assets/0022f23d-98ee-42a4-aa29-e4ffff6aed45" />
<img width="564" height="389" alt="image" src="https://github.com/user-attachments/assets/245bfda2-88f0-4c5b-9d6d-d85eb7d38125" />
<img width="292" height="262" alt="image" src="https://github.com/user-attachments/assets/ee5fc46f-d3a8-47f9-a6e4-c82273eed3cf" />
<img width="305" height="87" alt="image" src="https://github.com/user-attachments/assets/c24056f7-4563-4ddb-8a7d-015b1027027c" />
<img width="1264" height="255" alt="image" src="https://github.com/user-attachments/assets/afff6262-210b-4a76-8944-0b47db212f03" />
<img width="1197" height="238" alt="image" src="https://github.com/user-attachments/assets/36cfda19-b53d-4fdf-98cb-171e5ed82fcb" />
<img width="453" height="265" alt="image" src="https://github.com/user-attachments/assets/96d89805-7851-41cf-af03-ec647b26e124" />
<img width="439" height="89" alt="image" src="https://github.com/user-attachments/assets/accf5a9a-0ef8-4311-a2c8-314be02d9f1b" />
<img width="466" height="41" alt="image" src="https://github.com/user-attachments/assets/da837d23-a4a9-4239-aa56-b36e7d60f06c" />
<img width="233" height="45" alt="image" src="https://github.com/user-attachments/assets/c61fa565-f7a0-4fc7-9676-b250433d6416" />
<img width="251" height="48" alt="image" src="https://github.com/user-attachments/assets/fb847585-8d01-4bac-a7c7-ab5b93c0bc1d" />
<img width="843" height="610" alt="image" src="https://github.com/user-attachments/assets/f36ccc38-4423-41ef-89be-0b07c274f22e" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
