#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("Dataset:")
dataset = pd.read_csv('WaterPotability.csv') #reading the csv
dataset.dropna(inplace=True) #removing rows that is na or dont have value
dataset['Potability'] = dataset['Potability'].astype(int)
print(len(dataset))
print(dataset.head())


# In[14]:


columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']

scatter_matrix = pd.plotting.scatter_matrix(dataset[columns], figsize=(12,12)) #for layout

plt.tight_layout()
plt.show()


# In[15]:


dataset.describe()


# In[16]:


#for comparing of each columns to check for safe water consumption

features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
Potability = 'Potability'

colors = {0:'red', 1:'blue'}

for feature in features:
    plt.figure()
    plt.scatter(dataset[feature], dataset[Potability], c=[colors[label] for label in dataset[Potability]])
    plt.xlabel(feature)
    plt.ylabel(Potability)
    plt.title(f'{feature} vs Potability')
    
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

#Class for the model
#training and predicting the data set
class ModelEvaluator:
    def __init__(self, dataset_path, target_column_name, models): # Creating Attributes
        self.dataset_path = dataset_path
        self.target_column_name = target_column_name
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.models = models

#Reading and cleaning the data set, removing the data that don't have values or NA
    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)
        self.dataset.dropna(inplace=True)
        self.dataset[self.target_column_name] = self.dataset[self.target_column_name].astype(int)

#Function for preparing the dataset for training and testing
    def preprocess_data(self):
        X = self.dataset.drop(columns=[self.target_column_name]) #removing columns except the target column which is the potability
        y = self.dataset[self.target_column_name] #removing the values of the target column in dataset

        self.label_encoder = LabelEncoder() #for converting the string label into numbers
        y_encoded = self.label_encoder.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) #splitting the datasets into training and testing sets

#Function for scaling  the feature on the training and testing set
    def feature_scaling(self):
        sc_X = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test) #.transform is for mean and standard deviation
        
#Function for training and predicting the model
    def train_and_predict(self):
        results = {}
        for model_name, model in self.models.items():
            print(f"**Training and evaluating {model_name}**")
            y_pred = model.fit(self.X_train, self.y_train).predict(self.X_test) #using the current model using the training data set 
            results[model_name] = y_pred
            self.evaluate(model_name, y_pred)
        return results #for other classifier use outside this def or function.

#function for evaluating the performance of the model
    def evaluate(self, model_name, y_pred):
        mse = mean_squared_error(self.y_test, y_pred) #Calculating the mean squared error
        accuracy = accuracy_score(self.y_test, y_pred) #Calculating the accuracy
        f1 = f1_score(self.y_test, y_pred) #Calculating the model's prediction f1_score 
        print(f"{model_name} - Mean Squared Error: {mse}, Accuracy: {accuracy}, F1 Score: {f1}")

#For checking which statement is main and blocking the other statement
if __name__ == "__main__":
    dataset_path = 'WaterPotability.csv'
    target_column_name = 'Potability'

#Creating instances of the models
    knn_classifier = KNeighborsClassifier(n_neighbors=14, p=2, metric='euclidean')
    svm_classifier = SVC(kernel='linear', random_state=0)
    logistic_regression = LogisticRegression(random_state=0)

#Creating the dictionary of models
    models = {
        'KNN Classifier': knn_classifier,
        'SVM Classifier': svm_classifier,
        'Logistic Regression': logistic_regression,}

#Creating ModelEvaluator instance, preprocess data, feature scaling and evaluating models
    evaluator = ModelEvaluator(dataset_path, target_column_name, models)
    evaluator.load_dataset()
    evaluator.preprocess_data()
    evaluator.feature_scaling()

#Train and evaluate all models
    results = evaluator.train_and_predict()


# In[18]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#Creating Class for ROC
class ROCPlotter:
    def __init__(self, X_train, X_test, y_train, y_test): #Creating Attributes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classifiers = []

#Function for adding a new classifier to the existing classifier list
    def add_classifier(self, classifier, name):
        self.classifiers.append((classifier, name))

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'r--', label='Random')
        
        for classifier, name in self.classifiers: #using for loop for processing the training dataset
            classifier.fit(self.X_train, self.y_train)
            y_scores = classifier.predict_proba(self.X_test)[:, 1] #for predicting the probabilities of positive class
            fpr, tpr, _ = roc_curve(self.y_test, y_scores) #computing the false positive rate
            roc_auc = roc_auc_score(self.y_test, y_scores) #Calculating the AUC ROC
            plt.plot(fpr, tpr, label=f'{name} (area={roc_auc:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()
        
        
#For checking which statement is main and blocking the other statement
if __name__ == '__main__':    #Load and preprocess the dataset
    dataset_path = 'WaterPotability.csv'
    target_column_name = 'Potability'
    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)
    dataset['Potability'] = dataset['Potability'].astype(int)
    X = dataset.iloc[:, 0:9]
    y = dataset.iloc[:, 9]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

#Creating ROCPlotter and add a classifiers
    roc_plotter = ROCPlotter(X_train, X_test, y_train, y_test)

#Adding the Logistic Regression
    model_lr = LogisticRegression()
    roc_plotter.add_classifier(model_lr, 'Logistic Regression')

#Creating Decision Tree Classifier
    model_dt = DecisionTreeClassifier()
    roc_plotter.add_classifier(model_dt, 'Decision Tree Classifier')

#Plotting ROC curves for all classifiers
    roc_plotter.plot_roc_curves()


# In[ ]:




