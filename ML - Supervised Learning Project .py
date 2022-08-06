#!/usr/bin/env python
# coding: utf-8

# # Supervised Learning Project
# 
# ------
# 
# ## GOAL: Create a model to predict whether or not a customer will Churn .
# 
# ----
# 
# ## Part 0: Imports and Read in the Data

# In[250]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[251]:


df = pd.read_csv('Telco-Customer-Churn.csv')


# In[252]:


df.head()


# ## Part 1: Quick Data Check

# In[253]:


df.info()


# ## Statistical Summary

# In[254]:


df.describe()


# # Part 2:  Exploratory Data Analysis
# 
# ## General Feature Exploration
# 
# **TASK: Confirm that there are no NaN cells by displaying NaN values per feature column.**

# In[255]:


#Confirming that there are no NaN cells by displaying NaN values per feature column
df.isnull().sum()


# In[256]:


#Displaying the balance of the class labels (Churn) with a Count Plot.
sns.countplot(data=df, x='Churn')


# In[257]:


#Exploring the distrbution of TotalCharges between Churn categories with a Box Plot or Violin Plot

sns.boxplot(data=df, x='Churn', y='TotalCharges')


# In[258]:


#Creating a boxplot showing the distribution of TotalCharges per Contract type, also add in a hue coloring based on the Churn class

plt.figure(figsize=(12,6), dpi=200)
sns.boxplot(data=df, x='Contract', y='TotalCharges', hue='Churn')
plt.legend(loc=(1.05,0.5))


# In[259]:


#Creating Dummy variables for categorical features

corr_df = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 
 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'InternetService',
   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']])


# In[260]:


#Calculating Correlation of the above features to the target "Class" label

corr_df.corr()['Churn_Yes'].sort_values()[1:-1]


# In[261]:


#Creating BarPlot for correlation

plt.figure(figsize=(12,6), dpi=200)
corr_df.corr()['Churn_Yes'].sort_values()[1:-1].plot(kind='bar')
plt.title('Feature Correlation to Yes Churn')


# ---
# 
# # Part 3: Churn Analysis
# 
# **This section focuses on segementing customers based on their tenure, creating "cohorts", allowing us to examine differences between customer cohort segments.**

# In[262]:


# Analyzing unique Contract Types available

df['Contract'].unique()


# In[ ]:





# In[263]:


#Creating a histogram displaying the distribution of 'tenure' column, which is the amount of months a customer was or has been on a customer

plt.figure(figsize=(12,6))
sns.histplot(data=df, x='tenure', bins=50)


# In[264]:


#Creating histograms separated by two additional features, Churn and Contract

sns.displot(data=df, x='tenure',bins=70,row='Churn',col='Contract')


# In[265]:


#Creating a scatter plot of Total Charges versus Monthly Charges, and color hue by Churn
plt.figure(figsize=(12,6), dpi=200)
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn')


# ### Creating Cohorts based on Tenure and calculating the Churn rate (percentage that had Yes Churn) per cohort.

# In[266]:


#CODE HERE
churn_yes = 100 * len(df[df['Churn']=='Yes'])/len(df)
churn_yes


# In[267]:


churn_no = 100 * len(df[df['Churn']=='No'])/len(df)
churn_no


# In[268]:


df.groupby('tenure').apply(lambda df: 100 * len(df[df['Churn']=='Yes'])/len(df))


# ### From the above analysis, we find that the general trend is longer the tenure of the cohort, less is the churn rate

# In[269]:


#Displaying plot for Churn Rate per months of tenure
plt.figure(figsize=(12,6))
df.groupby('tenure').apply(lambda df: 100 * len(df[df['Churn']=='Yes'])/len(df)).plot()


# ### Broader Cohort Groups
# #### Based on the tenure column values, creating a new column called Tenure Cohort that creates 4 separate categories:**
#    * '0-12 Months'
#    * '12-24 Months'
#    * '24-48 Months'
#    * 'Over 48 Months'    

# In[270]:


# CODE HERE
def cohort(x):
    if x<13:
        return '0-12 months'
    elif x>12 and x<25:
        return '12-24 months'
    elif x>24 and x<49:
        return '24-48 months'
    elif x>48:
        return 'Over 48 months'

df['Tenure Cohort'] = df['tenure'].apply(cohort)


# In[271]:


df


# In[272]:


df[['tenure','Tenure Cohort']]


# In[273]:


#Creating a scatterplot of Total Charges versus Monthly Charts,colored by Tenure Cohort
plt.figure(figsize=(10,6), dpi=200)
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Tenure Cohort')


# In[274]:


#Creating a count plot showing the churn count per cohort
plt.figure(figsize=(10,6), dpi=200)
sns.countplot(data=df, x='Tenure Cohort', hue='Churn')


# In[275]:


#Creating a grid of Count Plots showing counts per Tenure Cohort, separated out by contract type and colored by the Churn hue

sns.catplot(data=df, x='Tenure Cohort', col='Contract', hue='Churn', kind='count')


# -----
# 
# # Part 4: Predictive Modeling
# 
# **Exploring various supervised learning models to make comparisons : 
# 1)Logistic Regression
# 2)K Nearest Neighbors
# 3)A Single Decision Tree
# 4)Random Forest
# 5)AdaBoost
# 6)Gradient Boosting**

# **Separating out the data into X features and Y label. Creating dummy variables where necessary**

# In[276]:


X = pd.get_dummies(df.drop(['Churn','customerID'], axis=1), drop_first=True)


# In[277]:


y = df['Churn']


# **Performing a train test split, holding out 10% of the data for testing**

# In[278]:


from sklearn.model_selection import train_test_split


# In[279]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# In[280]:


#Performing Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# ## 1)Logistic Regression

# In[281]:


from sklearn.linear_model import LogisticRegressionCV 


# In[282]:


log_model = LogisticRegressionCV()


# In[283]:


log_model.fit(scaled_X_train,y_train)


# **Reporting back your search's optimal parameters, specifically the C value**

# In[284]:


log_model.C_


# In[285]:


log_model.get_params()


# **Calculating Performance Metrics for Logistic Regression**

# In[286]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix


# In[287]:


y_pred = log_model.predict(scaled_X_test)


# In[288]:


confusion_matrix(y_test,y_pred)


# In[289]:


plot_confusion_matrix(log_model,scaled_X_test,y_test)


# In[290]:


print(classification_report(y_test,y_pred))


# ## 2)K Nearest Neighbor (KNN)

# **Creating a PipeLine that contains both a StandardScaler and a KNN model**

# In[291]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[292]:


scaler = StandardScaler()


# In[293]:


knn = KNeighborsClassifier()


# In[294]:


operations = [('scaler',scaler),('knn',knn)]


# In[295]:


from sklearn.pipeline import Pipeline


# In[296]:


pipe = Pipeline(operations)


# **Performing a grid-search with the pipeline to test various values of k and report back the best performing parameters.**

# In[297]:


from sklearn.model_selection import GridSearchCV


# In[298]:


k_values = list(range(1,30))


# In[299]:


param_grid = {'knn__n_neighbors': k_values}


# In[300]:


full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')


# In[301]:


full_cv_classifier.fit(X_train,y_train)


# **Getting the best parameters**

# In[302]:


full_cv_classifier.best_estimator_.get_params()


# In[303]:


#Plotting K values against Accuracy
scores = full_cv_classifier.cv_results_['mean_test_score']
plt.plot(k_values,scores,'o-')
plt.xlabel("K")
plt.ylabel("Accuracy")


# **Performance Metrics for K Nearest Neighbors**

# In[304]:


pred = full_cv_classifier.predict(X_test)


# In[305]:


confusion_matrix(y_test,pred)


# In[306]:


print(classification_report(y_test,pred))


# In[307]:


plot_confusion_matrix(full_cv_classifier, X_test, y_test)


# ## 3)Single Decision Tree

# In[308]:


from sklearn.tree import DecisionTreeClassifier


# In[309]:


model = DecisionTreeClassifier()


# In[310]:


from sklearn.model_selection import GridSearchCV


# In[311]:


grid_params = {'max_depth' : range(1,10)}


# In[312]:


grid_model = GridSearchCV(model, grid_params)


# In[313]:


grid_model.fit(scaled_X_train, y_train)


# **Finding the number of trees for which the model performed best**

# In[314]:


grid_model.best_estimator_


# **Using the value to create a new decision tree**

# In[315]:


latest_model = DecisionTreeClassifier(max_depth=5)


# In[316]:


latest_model.fit(scaled_X_train, y_train)


# **Calculating Performance Metrics for Single Decision Tree**

# In[317]:


from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score, confusion_matrix


# In[318]:


y_pred = latest_model.predict(scaled_X_test)


# In[319]:


confusion_matrix(y_test, y_pred)


# In[320]:


print(classification_report(y_test, y_pred))


# In[321]:


plot_confusion_matrix(latest_model, scaled_X_test, y_test)


# In[322]:


#Calculating Feature Importance
latest_model.feature_importances_


# **Plotting the feature Importance**

# In[323]:


plt.figure(figsize=(12,6), dpi=200)
pd.Series(index=X.columns, data = latest_model.feature_importances_).sort_values().plot(kind='bar')


# **Plotting the Decision Tree**

# In[324]:


from sklearn.tree import plot_tree
plt.figure(figsize=(50,25), dpi=200)
plot_tree(latest_model);


# ## 4)Random Forest
# 
# **Creating a Random Forest model and create a classification report and confusion matrix from its predicted results on the test set.**

# In[325]:


#CODE HERE
from sklearn.ensemble import RandomForestClassifier


# In[326]:


RFmodel = RandomForestClassifier()


# In[327]:


RFmodel.fit(scaled_X_train, y_train)


# In[328]:


y_pred_rf = RFmodel.predict(scaled_X_test)


# In[329]:


print(classification_report(y_test, y_pred_rf))


# In[330]:


plot_confusion_matrix(RFmodel, scaled_X_test, y_test)


# ## 5)Boosted Trees
# 
# **Using AdaBoost or Gradient Boosting to create a model and report back the classification report and plot a confusion matrix for its predicted results**

# In[331]:


#CODE HERE
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier


# In[332]:


ada = AdaBoostClassifier()


# In[333]:


gb = GradientBoostingClassifier()


# In[334]:


ada.fit(scaled_X_train, y_train)


# In[335]:


gb.fit(scaled_X_train, y_train)


# In[336]:


ada_y_pred = ada.predict(scaled_X_test)


# In[337]:


gb_y_pred = gb.predict(scaled_X_test)


# **Creating Classification Report for Ada Boosting**

# In[338]:


print(classification_report(ada_y_pred, y_test))


# **Creating Classification Report for Gradient Boosting**

# In[339]:


print(classification_report(gb_y_pred, y_test))


# **Creating Confusion Matrix for Ada Boosting**

# In[340]:


plot_confusion_matrix(ada, scaled_X_test, y_test)


# **Creating Confusion Matrix for Gradient Boosting**

# In[341]:


plot_confusion_matrix(gb, scaled_X_test, y_test)


# # With base models, we got best performance from Logistic Regression Model and an AdaBoostClassifier with an accuracy of about 83%
