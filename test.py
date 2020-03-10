# DataSet :
# https://www.kaggle.com/crowdflower/twitter-user-gender-classification

import pandas as pd


import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams



# Loading Dataset
wholeData = pd.read_csv("./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data = pd.DataFrame(wholeData)

print("Before CustomerID drop\n",data.head())
#Before converting gender data into numirecal
# print(data['gender'])
# print(data.dtypes)


# exploratory data analysis :1
# Checking null values

# print(data.isnull().sum())

# We can below methods to replace null values with sum but we are not doing it as we dont have any null values

import numpy as np
# data = data.select_dtypes(include=[np.number]).interpolate().dropna()

# Dropping the customerID column
data.drop(columns={'customerID'},inplace=True)
print("After CustomerID drop\n",data.head())
# exploratory data analysis :2
# Converting the categorical data to the numerical data
telcom_df =  data.iloc[:,:19]
print("Telcom data [:19]\n",telcom_df.head())

telcom_df['gender']=telcom_df.gender.apply(lambda x: 1 if x=='Male' else 0)
telcom_df['Partner']=telcom_df.Partner.apply(lambda x: 1 if x=='Yes' else 0)
telcom_df['Dependents']=telcom_df.Dependents.apply(lambda x: 1 if x=='Yes' else 0)
telcom_df['PhoneService']=telcom_df.PhoneService.apply(lambda x: 1 if x=='Yes' else 0)
telcom_df['PaperlessBilling']=telcom_df.PaperlessBilling.apply(lambda x: 1 if x=='Yes' else 0)



# Dropping the columns which have dummy values
telcom_df.drop(columns={'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod','TotalCharges'},inplace=True)
print("After removing dummy vales\n",telcom_df.head())

#After converting gender data into numirecal
# print(telcom_df['gender'])

# exploratory data analysis :3
# removing of correlated features.
# Finding the correlation and displaying the heatmap plot

telcom_df.corr()
print(telcom_df['gender'].sort_values(ascending=True)[:5])

# sns.heatmap(telcom_df, annot=True, cmap=plt.cm.Reds)

# plt.show()

# Dropping the columns which have very less correlation
telcom_df.drop(columns={'MonthlyCharges',
                        'PaperlessBilling',
                        'PhoneService',
                        },inplace=True)
print("After Drooping less Correlation\n",telcom_df)

#Preprocessing data
x=telcom_df.drop('gender',axis=1)
print("X values\n",x)

y=telcom_df['gender']
print("Y values\n",y)
from sklearn import model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x,y,test_size=0.2,random_state=0)# 80% training and 20% test
print("X-Train\n",X_train.dtypes)
print("Y-Train\n",Y_train.dtypes)
print("X-Test\n",X_test.dtypes)
print("Y-Test\n",Y_test.dtypes)

#-----------Prediction Using Naive Bayes Classifier ----------#
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = model.predict(X_test)


#----------Evaluating the model -------------#
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Naive Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
# print("Naive classification_report\n",metrics.classification_report(Y_test,Y_pred))

# -----------Prediction Using SVM (Support Vector Machines)----------#

from sklearn import svm
from sklearn.model_selection import KFold
accuracy = []

model = svm.SVC(kernel='linear')

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
accuracy.append(metrics.accuracy_score(Y_test, y_pred))

# print("Mean Accuracy Score: ", np.mean(accuracy))
print("SVM Accuracy score: ", metrics.accuracy_score(Y_test, y_pred))


#-----------Prediction Using k-nearest neighbors ----------#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) #set K neighbor as 3
knn.fit(X_train,Y_train)
predicted_y = knn.predict(X_test)
acc_knn= round(knn.score(X_train, Y_train) * 100, 2)
# print("KNN accuracy is:",acc_knn)
print("KNN accuracy according to K=3 is :",knn.score(X_test,Y_test))