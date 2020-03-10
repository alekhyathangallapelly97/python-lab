import pandas as pd
import numpy as np

#Importing data
adults_data = pd.read_csv('./dataset/dataset_183_adult.csv')
training_data = adults_data.drop("class",axis=1)
label = adults_data['class']

#Counting the null values
print("Number of null values present in the data:\n", format(adults_data.isnull().sum()))

#Eliminating null values
adults_data.dropna(axis = 0, inplace= True)

#Encoding the categorial feature
data_binary = pd.get_dummies(adults_data) #Convert categorical variable into dummy/indicator variables.
data_binary.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_binary,label)
performance_score = []

# Gaussin Naive Bayes classification
from sklearn.naive_bayes import GaussianNB
GNB_model = GaussianNB()
GNB_model.fit(x_train,y_train)
train_score = GNB_model.score(x_train,y_train)
test_score = GNB_model.score(x_test,y_test)
print(f'Training score of Gaussin Naive Bayes- {train_score} - Test score of Gaussin Naive Bayes- {test_score}')
performance_score.append({'Model':'Gaussian Naive Bayes', 'Training score':train_score, 'Testing score':test_score})

#KNN classification
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train,y_train)
knn_model.score(x_train,y_train)
train_score = knn_model.score(x_train,y_train)
test_score = knn_model.score(x_test,y_test)
print(f'Training score of KNN Classification- {train_score} - Test score of KNN Classification- {test_score}')
performance_score.append({'Model':'KNN', 'Training score':train_score, 'Testing score':test_score})

#SVM classification

from sklearn import svm

svc_model = svm.SVC(kernel='linear')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_binary,label)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
svc_model.fit(x_train_scaled,y_train)
train_score = svc_model.score(x_train_scaled,y_train)
test_score = svc_model.score(x_test_scaled, y_test)
print(f'Training score of SVM Classification- {train_score} - Test score of SVM Classification- {test_score}')
performance_score.append({'Model':'SVM Classification', 'Training score':train_score, 'Testing score':test_score})