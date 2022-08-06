# Supervised Learning Project

------

## GOAL: Create a model to predict whether or not a customer will Churn .

----

## Part 0: Imports and Read in the Data


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('Telco-Customer-Churn.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



## Part 1: Quick Data Check


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7032 entries, 0 to 7031
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   customerID        7032 non-null   object 
     1   gender            7032 non-null   object 
     2   SeniorCitizen     7032 non-null   int64  
     3   Partner           7032 non-null   object 
     4   Dependents        7032 non-null   object 
     5   tenure            7032 non-null   int64  
     6   PhoneService      7032 non-null   object 
     7   MultipleLines     7032 non-null   object 
     8   InternetService   7032 non-null   object 
     9   OnlineSecurity    7032 non-null   object 
     10  OnlineBackup      7032 non-null   object 
     11  DeviceProtection  7032 non-null   object 
     12  TechSupport       7032 non-null   object 
     13  StreamingTV       7032 non-null   object 
     14  StreamingMovies   7032 non-null   object 
     15  Contract          7032 non-null   object 
     16  PaperlessBilling  7032 non-null   object 
     17  PaymentMethod     7032 non-null   object 
     18  MonthlyCharges    7032 non-null   float64
     19  TotalCharges      7032 non-null   float64
     20  Churn             7032 non-null   object 
    dtypes: float64(2), int64(2), object(17)
    memory usage: 1.1+ MB
    

## Statistical Summary


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeniorCitizen</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.162400</td>
      <td>32.421786</td>
      <td>64.798208</td>
      <td>2283.300441</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.368844</td>
      <td>24.545260</td>
      <td>30.085974</td>
      <td>2266.771362</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>18.250000</td>
      <td>18.800000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>35.587500</td>
      <td>401.450000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>70.350000</td>
      <td>1397.475000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>55.000000</td>
      <td>89.862500</td>
      <td>3794.737500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>72.000000</td>
      <td>118.750000</td>
      <td>8684.800000</td>
    </tr>
  </tbody>
</table>
</div>



# Part 2:  Exploratory Data Analysis

## General Feature Exploration

**TASK: Confirm that there are no NaN cells by displaying NaN values per feature column.**


```python
#Confirming that there are no NaN cells by displaying NaN values per feature column
df.isnull().sum()
```




    customerID          0
    gender              0
    SeniorCitizen       0
    Partner             0
    Dependents          0
    tenure              0
    PhoneService        0
    MultipleLines       0
    InternetService     0
    OnlineSecurity      0
    OnlineBackup        0
    DeviceProtection    0
    TechSupport         0
    StreamingTV         0
    StreamingMovies     0
    Contract            0
    PaperlessBilling    0
    PaymentMethod       0
    MonthlyCharges      0
    TotalCharges        0
    Churn               0
    dtype: int64




```python
#Displaying the balance of the class labels (Churn) with a Count Plot.
sns.countplot(data=df, x='Churn')
```




    <AxesSubplot:xlabel='Churn', ylabel='count'>




    
![png](output_10_1.png)
    



```python
#Exploring the distrbution of TotalCharges between Churn categories with a Box Plot or Violin Plot

sns.boxplot(data=df, x='Churn', y='TotalCharges')
```




    <AxesSubplot:xlabel='Churn', ylabel='TotalCharges'>




    
![png](output_11_1.png)
    



```python
#Creating a boxplot showing the distribution of TotalCharges per Contract type, also add in a hue coloring based on the Churn class

plt.figure(figsize=(12,6), dpi=200)
sns.boxplot(data=df, x='Contract', y='TotalCharges', hue='Churn')
plt.legend(loc=(1.05,0.5))
```




    <matplotlib.legend.Legend at 0x2317fa4a790>




    
![png](output_12_1.png)
    



```python
#Creating Dummy variables for categorical features

corr_df = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 
 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'InternetService',
   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']])
```


```python
#Calculating Correlation of the above features to the target "Class" label

corr_df.corr()['Churn_Yes'].sort_values()[1:-1]
```




    Contract_Two year                         -0.301552
    DeviceProtection_No internet service      -0.227578
    StreamingMovies_No internet service       -0.227578
    StreamingTV_No internet service           -0.227578
    InternetService_No                        -0.227578
    TechSupport_No internet service           -0.227578
    OnlineBackup_No internet service          -0.227578
    OnlineSecurity_No internet service        -0.227578
    PaperlessBilling_No                       -0.191454
    Contract_One year                         -0.178225
    OnlineSecurity_Yes                        -0.171270
    TechSupport_Yes                           -0.164716
    Dependents_Yes                            -0.163128
    Partner_Yes                               -0.149982
    PaymentMethod_Credit card (automatic)     -0.134687
    InternetService_DSL                       -0.124141
    PaymentMethod_Bank transfer (automatic)   -0.118136
    PaymentMethod_Mailed check                -0.090773
    OnlineBackup_Yes                          -0.082307
    DeviceProtection_Yes                      -0.066193
    MultipleLines_No                          -0.032654
    MultipleLines_No phone service            -0.011691
    PhoneService_No                           -0.011691
    gender_Male                               -0.008545
    gender_Female                              0.008545
    PhoneService_Yes                           0.011691
    MultipleLines_Yes                          0.040033
    StreamingMovies_Yes                        0.060860
    StreamingTV_Yes                            0.063254
    StreamingTV_No                             0.128435
    StreamingMovies_No                         0.130920
    Partner_No                                 0.149982
    SeniorCitizen                              0.150541
    Dependents_No                              0.163128
    PaperlessBilling_Yes                       0.191454
    DeviceProtection_No                        0.252056
    OnlineBackup_No                            0.267595
    PaymentMethod_Electronic check             0.301455
    InternetService_Fiber optic                0.307463
    TechSupport_No                             0.336877
    OnlineSecurity_No                          0.342235
    Contract_Month-to-month                    0.404565
    Name: Churn_Yes, dtype: float64




```python
#Creating BarPlot for correlation

plt.figure(figsize=(12,6), dpi=200)
corr_df.corr()['Churn_Yes'].sort_values()[1:-1].plot(kind='bar')
plt.title('Feature Correlation to Yes Churn')
```




    Text(0.5, 1.0, 'Feature Correlation to Yes Churn')




    
![png](output_15_1.png)
    


---

# Part 3: Churn Analysis

**This section focuses on segementing customers based on their tenure, creating "cohorts", allowing us to examine differences between customer cohort segments.**


```python
# Analyzing unique Contract Types available

df['Contract'].unique()
```




    array(['Month-to-month', 'One year', 'Two year'], dtype=object)




```python

```


```python
#Creating a histogram displaying the distribution of 'tenure' column, which is the amount of months a customer was or has been on a customer

plt.figure(figsize=(12,6))
sns.histplot(data=df, x='tenure', bins=50)
```




    <AxesSubplot:xlabel='tenure', ylabel='Count'>




    
![png](output_19_1.png)
    



```python
#Creating histograms separated by two additional features, Churn and Contract

sns.displot(data=df, x='tenure',bins=70,row='Churn',col='Contract')
```




    <seaborn.axisgrid.FacetGrid at 0x23185470c70>




    
![png](output_20_1.png)
    



```python
#Creating a scatter plot of Total Charges versus Monthly Charges, and color hue by Churn
plt.figure(figsize=(12,6), dpi=200)
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn')
```




    <AxesSubplot:xlabel='MonthlyCharges', ylabel='TotalCharges'>




    
![png](output_21_1.png)
    


### Creating Cohorts based on Tenure and calculating the Churn rate (percentage that had Yes Churn) per cohort.


```python
#CODE HERE
churn_yes = 100 * len(df[df['Churn']=='Yes'])/len(df)
churn_yes

```




    26.57849829351536




```python
churn_no = 100 * len(df[df['Churn']=='No'])/len(df)
churn_no
```




    73.42150170648465




```python
df.groupby('tenure').apply(lambda df: 100 * len(df[df['Churn']=='Yes'])/len(df))
```




    tenure
    1     61.990212
    2     51.680672
    3     47.000000
    4     47.159091
    5     48.120301
            ...    
    68     9.000000
    69     8.421053
    70     9.243697
    71     3.529412
    72     1.657459
    Length: 72, dtype: float64



### From the above analysis, we find that the general trend is longer the tenure of the cohort, less is the churn rate


```python
#Displaying plot for Churn Rate per months of tenure
plt.figure(figsize=(12,6))
df.groupby('tenure').apply(lambda df: 100 * len(df[df['Churn']=='Yes'])/len(df)).plot()
```




    <AxesSubplot:xlabel='tenure'>




    
![png](output_27_1.png)
    


### Broader Cohort Groups
#### Based on the tenure column values, creating a new column called Tenure Cohort that creates 4 separate categories:**
   * '0-12 Months'
   * '12-24 Months'
   * '24-48 Months'
   * 'Over 48 Months'    


```python
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
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>Tenure Cohort</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>No</td>
      <td>24-48 months</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
      <td>24-48 months</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7027</th>
      <td>6840-RESVB</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>24</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>84.80</td>
      <td>1990.50</td>
      <td>No</td>
      <td>12-24 months</td>
    </tr>
    <tr>
      <th>7028</th>
      <td>2234-XADUH</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>72</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>103.20</td>
      <td>7362.90</td>
      <td>No</td>
      <td>Over 48 months</td>
    </tr>
    <tr>
      <th>7029</th>
      <td>4801-JZAZL</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>11</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.60</td>
      <td>346.45</td>
      <td>No</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>7030</th>
      <td>8361-LTMKD</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>4</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>74.40</td>
      <td>306.60</td>
      <td>Yes</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>7031</th>
      <td>3186-AJIEK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>66</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>105.65</td>
      <td>6844.50</td>
      <td>No</td>
      <td>Over 48 months</td>
    </tr>
  </tbody>
</table>
<p>7032 rows × 22 columns</p>
</div>




```python
df[['tenure','Tenure Cohort']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tenure</th>
      <th>Tenure Cohort</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>24-48 months</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>24-48 months</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7027</th>
      <td>24</td>
      <td>12-24 months</td>
    </tr>
    <tr>
      <th>7028</th>
      <td>72</td>
      <td>Over 48 months</td>
    </tr>
    <tr>
      <th>7029</th>
      <td>11</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>7030</th>
      <td>4</td>
      <td>0-12 months</td>
    </tr>
    <tr>
      <th>7031</th>
      <td>66</td>
      <td>Over 48 months</td>
    </tr>
  </tbody>
</table>
<p>7032 rows × 2 columns</p>
</div>




```python
#Creating a scatterplot of Total Charges versus Monthly Charts,colored by Tenure Cohort
plt.figure(figsize=(10,6), dpi=200)
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Tenure Cohort')
```




    <AxesSubplot:xlabel='MonthlyCharges', ylabel='TotalCharges'>




    
![png](output_32_1.png)
    



```python
#Creating a count plot showing the churn count per cohort
plt.figure(figsize=(10,6), dpi=200)
sns.countplot(data=df, x='Tenure Cohort', hue='Churn')
```




    <AxesSubplot:xlabel='Tenure Cohort', ylabel='count'>




    
![png](output_33_1.png)
    



```python
#Creating a grid of Count Plots showing counts per Tenure Cohort, separated out by contract type and colored by the Churn hue

sns.catplot(data=df, x='Tenure Cohort', col='Contract', hue='Churn', kind='count')
```




    <seaborn.axisgrid.FacetGrid at 0x2318708bc10>




    
![png](output_34_1.png)
    


-----

# Part 4: Predictive Modeling

**Exploring various supervised learning models to make comparisons : 
1)Logistic Regression
2)K Nearest Neighbors
3)A Single Decision Tree
4)Random Forest
5)AdaBoost
6)Gradient Boosting**

**Separating out the data into X features and Y label. Creating dummy variables where necessary**


```python
X = pd.get_dummies(df.drop(['Churn','customerID'], axis=1), drop_first=True)
```


```python
y = df['Churn']
```

**Performing a train test split, holding out 10% of the data for testing**


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
```


```python
#Performing Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
```

## 1)Logistic Regression


```python
from sklearn.linear_model import LogisticRegressionCV 
```


```python
log_model = LogisticRegressionCV()
```


```python
log_model.fit(scaled_X_train,y_train)
```




    LogisticRegressionCV()



**Reporting back your search's optimal parameters, specifically the C value**


```python
log_model.C_
```




    array([0.04641589])




```python
log_model.get_params()
```




    {'Cs': 10,
     'class_weight': None,
     'cv': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1.0,
     'l1_ratios': None,
     'max_iter': 100,
     'multi_class': 'auto',
     'n_jobs': None,
     'penalty': 'l2',
     'random_state': None,
     'refit': True,
     'scoring': None,
     'solver': 'lbfgs',
     'tol': 0.0001,
     'verbose': 0}



**Calculating Performance Metrics for Logistic Regression**


```python
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
```


```python
y_pred = log_model.predict(scaled_X_test)
```


```python
confusion_matrix(y_test,y_pred)
```




    array([[509,  48],
           [ 73,  74]], dtype=int64)




```python
plot_confusion_matrix(log_model,scaled_X_test,y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x231871c84f0>




    
![png](output_54_1.png)
    



```python
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
              No       0.87      0.91      0.89       557
             Yes       0.61      0.50      0.55       147
    
        accuracy                           0.83       704
       macro avg       0.74      0.71      0.72       704
    weighted avg       0.82      0.83      0.82       704
    
    

## 2)K Nearest Neighbor (KNN)

**Creating a PipeLine that contains both a StandardScaler and a KNN model**


```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
```


```python
scaler = StandardScaler()
```


```python
knn = KNeighborsClassifier()
```


```python
operations = [('scaler',scaler),('knn',knn)]
```


```python
from sklearn.pipeline import Pipeline
```


```python
pipe = Pipeline(operations)
```

**Performing a grid-search with the pipeline to test various values of k and report back the best performing parameters.**


```python
from sklearn.model_selection import GridSearchCV
```


```python
k_values = list(range(1,30))
```


```python
param_grid = {'knn__n_neighbors': k_values}
```


```python
full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')
```


```python
full_cv_classifier.fit(X_train,y_train)
```




    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                           ('knn', KNeighborsClassifier())]),
                 param_grid={'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                                  12, 13, 14, 15, 16, 17, 18, 19,
                                                  20, 21, 22, 23, 24, 25, 26, 27,
                                                  28, 29]},
                 scoring='accuracy')



**Getting the best parameters**


```python
full_cv_classifier.best_estimator_.get_params()
```




    {'memory': None,
     'steps': [('scaler', StandardScaler()),
      ('knn', KNeighborsClassifier(n_neighbors=28))],
     'verbose': False,
     'scaler': StandardScaler(),
     'knn': KNeighborsClassifier(n_neighbors=28),
     'scaler__copy': True,
     'scaler__with_mean': True,
     'scaler__with_std': True,
     'knn__algorithm': 'auto',
     'knn__leaf_size': 30,
     'knn__metric': 'minkowski',
     'knn__metric_params': None,
     'knn__n_jobs': None,
     'knn__n_neighbors': 28,
     'knn__p': 2,
     'knn__weights': 'uniform'}




```python
#Plotting K values against Accuracy
scores = full_cv_classifier.cv_results_['mean_test_score']
plt.plot(k_values,scores,'o-')
plt.xlabel("K")
plt.ylabel("Accuracy")
```




    Text(0, 0.5, 'Accuracy')




    
![png](output_72_1.png)
    


**Performance Metrics for K Nearest Neighbors**


```python
pred = full_cv_classifier.predict(X_test)
```


```python
confusion_matrix(y_test,pred)
```




    array([[498,  59],
           [ 74,  73]], dtype=int64)




```python
print(classification_report(y_test,pred))
```

                  precision    recall  f1-score   support
    
              No       0.87      0.89      0.88       557
             Yes       0.55      0.50      0.52       147
    
        accuracy                           0.81       704
       macro avg       0.71      0.70      0.70       704
    weighted avg       0.80      0.81      0.81       704
    
    


```python
plot_confusion_matrix(full_cv_classifier, X_test, y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2317ed00760>




    
![png](output_77_1.png)
    


## 3)Single Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
model = DecisionTreeClassifier()
```


```python
from sklearn.model_selection import GridSearchCV
```


```python
grid_params = {'max_depth' : range(1,10)}
```


```python
grid_model = GridSearchCV(model, grid_params)
```


```python
grid_model.fit(scaled_X_train, y_train)
```




    GridSearchCV(estimator=DecisionTreeClassifier(),
                 param_grid={'max_depth': range(1, 10)})



**Finding the number of trees for which the model performed best**


```python
grid_model.best_estimator_
```




    DecisionTreeClassifier(max_depth=5)



**Using the value to create a new decision tree**


```python
latest_model = DecisionTreeClassifier(max_depth=5)
```


```python
latest_model.fit(scaled_X_train, y_train)
```




    DecisionTreeClassifier(max_depth=5)



**Calculating Performance Metrics for Single Decision Tree**


```python
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score, confusion_matrix
```


```python
y_pred = latest_model.predict(scaled_X_test)
```


```python
confusion_matrix(y_test, y_pred)
```




    array([[498,  59],
           [ 83,  64]], dtype=int64)




```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
              No       0.86      0.89      0.88       557
             Yes       0.52      0.44      0.47       147
    
        accuracy                           0.80       704
       macro avg       0.69      0.66      0.67       704
    weighted avg       0.79      0.80      0.79       704
    
    


```python
plot_confusion_matrix(latest_model, scaled_X_test, y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x23185496070>




    
![png](output_95_1.png)
    



```python
#Calculating Feature Importance
latest_model.feature_importances_
```




    array([0.01018467, 0.45223938, 0.02467061, 0.04460289, 0.        ,
           0.        , 0.        , 0.        , 0.00754192, 0.00799877,
           0.34022334, 0.        , 0.        , 0.00567638, 0.        ,
           0.00578629, 0.        , 0.        , 0.0284796 , 0.00852294,
           0.        , 0.        , 0.        , 0.        , 0.00962181,
           0.02332128, 0.        , 0.        , 0.03113012, 0.        ,
           0.        , 0.        , 0.        ])



**Plotting the feature Importance**


```python
plt.figure(figsize=(12,6), dpi=200)
pd.Series(index=X.columns, data = latest_model.feature_importances_).sort_values().plot(kind='bar')
```




    <AxesSubplot:>




    
![png](output_98_1.png)
    


**Plotting the Decision Tree**


```python
from sklearn.tree import plot_tree
plt.figure(figsize=(50,25), dpi=200)
plot_tree(latest_model);
```


    
![png](output_100_0.png)
    


## 4)Random Forest

**Creating a Random Forest model and create a classification report and confusion matrix from its predicted results on the test set.**


```python
#CODE HERE
from sklearn.ensemble import RandomForestClassifier
```


```python
RFmodel = RandomForestClassifier()
```


```python
RFmodel.fit(scaled_X_train, y_train)
```




    RandomForestClassifier()




```python
y_pred_rf = RFmodel.predict(scaled_X_test)
```


```python
print(classification_report(y_test, y_pred_rf))
```

                  precision    recall  f1-score   support
    
              No       0.86      0.88      0.87       557
             Yes       0.50      0.46      0.48       147
    
        accuracy                           0.79       704
       macro avg       0.68      0.67      0.68       704
    weighted avg       0.79      0.79      0.79       704
    
    


```python
plot_confusion_matrix(RFmodel, scaled_X_test, y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2317ece2520>




    
![png](output_107_1.png)
    


## 5)Boosted Trees

**Using AdaBoost or Gradient Boosting to create a model and report back the classification report and plot a confusion matrix for its predicted results**


```python
#CODE HERE
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
```


```python
ada = AdaBoostClassifier()
```


```python
gb = GradientBoostingClassifier()
```


```python
ada.fit(scaled_X_train, y_train)
```




    AdaBoostClassifier()




```python
gb.fit(scaled_X_train, y_train)
```




    GradientBoostingClassifier()




```python
ada_y_pred = ada.predict(scaled_X_test)
```


```python
gb_y_pred = gb.predict(scaled_X_test)
```

**Creating Classification Report for Ada Boosting**


```python
print(classification_report(ada_y_pred, y_test))
```

                  precision    recall  f1-score   support
    
              No       0.90      0.88      0.89       571
             Yes       0.54      0.60      0.57       133
    
        accuracy                           0.83       704
       macro avg       0.72      0.74      0.73       704
    weighted avg       0.84      0.83      0.83       704
    
    

**Creating Classification Report for Gradient Boosting**


```python
print(classification_report(gb_y_pred, y_test))
```

                  precision    recall  f1-score   support
    
              No       0.90      0.87      0.89       577
             Yes       0.50      0.57      0.53       127
    
        accuracy                           0.82       704
       macro avg       0.70      0.72      0.71       704
    weighted avg       0.83      0.82      0.82       704
    
    

**Creating Confusion Matrix for Ada Boosting**


```python
plot_confusion_matrix(ada, scaled_X_test, y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x23188ac0be0>




    
![png](output_121_1.png)
    


**Creating Confusion Matrix for Gradient Boosting**


```python
plot_confusion_matrix(gb, scaled_X_test, y_test)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x231889b5130>




    
![png](output_123_1.png)
    


# With base models, we got best performance from Logistic Regression Model and an AdaBoostClassifier with an accuracy of about 83%
