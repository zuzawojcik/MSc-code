#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_roc_curve, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.svm import NuSVC
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


# In[2]:


dataframe = pd.read_csv('datafile.csv')
dataframe


# In[3]:


dataframe = dataframe.drop(['Unnamed: 0', 'StudyArm'], axis=1)


# In[4]:


admissions = []
for i in dataframe['NoofAdmissions']:
    if i == 0:
        admissions.append(0)
    else: 
        admissions.append(1)
        
dataframe['admissions'] = admissions

triage = []
for i in dataframe['NoofTriageEvents']:
    if i == 0:
        triage.append(0)
    else: 
        triage.append(1)
        
dataframe['triage'] = triage

chemo_change = []
for i in dataframe['ChangestoChemo']:
    if i == 0:
        chemo_change.append(0)
    else: 
        chemo_change.append(1)
        
dataframe['chemo_change'] = chemo_change
dataframe = dataframe.drop(['NoofTriageEvents','ChangestoChemo','NoofAdmissions'], axis=1)
dataframe


# In[7]:


for column in dataframe.columns:
    print(dataframe[column].value_counts())


# In[26]:


import statistics
dataframe.mean(numeric_only=True)


# In[22]:


dataframe.isna().sum()


# In[ ]:


data_all = dataframe.dropna(subset=['C30_Appetite_0'])
data_clinical = dataframe.iloc[:, [0,1,2,3,4,5,6,31,32,33]]
data_proms = dataframe.drop (['DiseaseSite', 'Sex', 'PreviousChemo', 'AgeStudyEntry', 'PrimaryorMet',
                              'Comorbidities', 'DaysonStudy'], axis=1)
data_C30 = dataframe.drop (['QoLEQ5DMob', 'QoLEQ5DSelCar', 'QoLEQ5DUsuAct', 'QoLEQ5DPain','QoLEQ5DAnxDep',
                            'PhysicalWB_Baseline', 'SocialWB_Baseline', 'EmotionalWB_Baseline',
                            'FunctionalWB_Baseline'], axis=1)
data_5D = dataframe.drop(['C30_Appetite_0', 'C30_Dyspnoea_0', 'C30_Pain_0', 'C30_Fatigue_0',
                          'C30_NauseaVom_0', 'C30_Const_0', 'C30_Diarr_0', 'C30_Financ_0',
                          'C30_GlobalHealth_0', 'C30_Cognitive_0', 'C30_Sleep_0',
                          'C30_Emotional_0', 'C30_Physical_0', 'C30_Role_0', 'C30_Social_0','PhysicalWB_Baseline', 
                          'SocialWB_Baseline', 'EmotionalWB_Baseline',
                            'FunctionalWB_Baseline'], axis = 1)
data_fact = dataframe.drop(['C30_Appetite_0', 'C30_Dyspnoea_0', 'C30_Pain_0', 'C30_Fatigue_0',
                          'C30_NauseaVom_0', 'C30_Const_0', 'C30_Diarr_0', 'C30_Financ_0',
                          'C30_GlobalHealth_0', 'C30_Cognitive_0', 'C30_Sleep_0',
                          'C30_Emotional_0', 'C30_Physical_0', 'C30_Role_0', 'C30_Social_0', 'QoLEQ5DMob', 
                            'QoLEQ5DSelCar', 'QoLEQ5DUsuAct', 'QoLEQ5DPain','QoLEQ5DAnxDep'], axis = 1)
data_proms = data_proms.dropna(subset=['C30_Appetite_0'])
data_C30 = data_C30.dropna(subset=['C30_Appetite_0'])


# In[ ]:


data_5D.columns


# In[ ]:


imputer = KNNImputer(n_neighbors=5)
data_all = pd.DataFrame(imputer.fit_transform(data_all),columns = data_all.columns)
data_clinical = pd.DataFrame(imputer.fit_transform(data_clinical),columns = data_clinical.columns)
data_proms = pd.DataFrame(imputer.fit_transform(data_proms),columns = data_proms.columns)
data_C30 = pd.DataFrame(imputer.fit_transform(data_C30),columns = data_C30.columns)
data_5D = pd.DataFrame(imputer.fit_transform(data_5D),columns = data_5D.columns)
data_fact = pd.DataFrame(imputer.fit_transform(data_fact),columns = data_fact.columns)
print (len(data_all), len(data_all.columns))
print (len(data_clinical), len(data_clinical.columns))
print (len(data_proms), len(data_proms.columns))
print (len(data_C30), len(data_C30.columns))
print (len(data_5D), len(data_5D.columns))
print (len(data_fact), len(data_fact.columns))


# In[ ]:


correlated_features = set()
correlation_matrix = data_all.corr()
threshold = 0.5

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features)


# In[ ]:


corr_data = data_all.filter(items=correlated_features)
cm = corr_data.corr()
s = cm.unstack()
so = s.sort_values()
so = so.to_frame().T


# In[ ]:


so


# In[ ]:


so = so.drop(columns=so.columns[(so == 1.0).any()])


# In[ ]:


so = so.drop(columns=so.columns[(abs(so) < 0.5).any()])


# In[ ]:


so.columns


# In[ ]:


data_all = data_all.drop(['C30_Physical_0', 'C30_Role_0','C30_GlobalHealth_0', 'C30_Emotional_0', 'C30_Fatigue_0','QoLEQ5DPain', 'QoLEQ5DMob','QoLEQ5DUsuAct', 'C30_Social_0'],axis=1)


# In[ ]:


correlated_features = set()
correlation_matrix = data_clinical.corr()
threshold = 0.50

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features) #sex with disease site 


# In[ ]:


data_clinical = data_clinical.drop(['Sex'], axis=1)


# In[ ]:


correlated_features = set()
correlation_matrix = data_proms.corr()
threshold = 0.50

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features)


# In[ ]:


corr_data = data_proms.filter(items=correlated_features)
cm = corr_data.corr()
s = cm.unstack()
so = s.sort_values()
so = so.to_frame().T


# In[ ]:


so


# In[ ]:


so = so.drop(columns=so.columns[(so == 1.0).any()])
so = so.drop(columns=so.columns[(abs(so) < 0.5).any()])
so.columns


# In[ ]:


data_proms = data_proms.drop(['C30_Physical_0', 'C30_Role_0','C30_GlobalHealth_0', 'C30_Emotional_0', 'C30_Fatigue_0','QoLEQ5DPain', 'QoLEQ5DMob','QoLEQ5DUsuAct', 'C30_Social_0'],axis=1)


# In[ ]:


correlated_features = set()
correlation_matrix = data_C30.corr()
threshold = 0.50

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features)


# In[ ]:


corr_data = data_C30.filter(items=correlated_features)
cm = corr_data.corr()
s = cm.unstack()
so = s.sort_values()
so = so.to_frame().T


# In[ ]:


so


# In[ ]:


so = so.drop(columns=so.columns[(so == 1.0).any()])
so = so.drop(columns=so.columns[(abs(so) < 0.5).any()])
so.columns


# In[ ]:


data_C30 = data_C30.drop(['C30_Fatigue_0','C30_Role_0','C30_Social_0','C30_GlobalHealth_0'], axis=1)


# In[ ]:


correlated_features = set()
correlation_matrix = data_5D.corr()
threshold = 0.50

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features)


# In[ ]:


corr_data = data_5D.filter(items=correlated_features)
cm = corr_data.corr()
s = cm.unstack()
so = s.sort_values()
so = so.to_frame().T


# In[ ]:


so = so.drop(columns=so.columns[(so == 1.0).any()])
so = so.drop(columns=so.columns[(abs(so) < 0.5).any()])
so.columns


# In[ ]:


data_5D = data_5D.drop(['QoLEQ5DPain'], axis=1)


# In[ ]:


correlated_features = set()
correlation_matrix = data_fact.corr()
threshold = 0.50

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features)


# In[ ]:


data_fact = data_fact.drop(['Sex', 'FunctionalWB_Baseline'], axis=1)


# In[ ]:


print (len(data_all), len(data_all.columns))
print (len(data_clinical), len(data_clinical.columns))
print (len(data_proms), len(data_proms.columns))
print (len(data_C30), len(data_C30.columns))
print (len(data_5D), len(data_5D.columns))
print (len(data_fact), len(data_fact.columns))


# In[ ]:


print(data_all['admissions'].value_counts())
print(data_all['triage'].value_counts())
print(data_all['chemo_change'].value_counts())
print(data_clinical['admissions'].value_counts())
print(data_clinical['triage'].value_counts())
print(data_clinical['chemo_change'].value_counts())
print(data_proms['admissions'].value_counts())
print(data_proms['triage'].value_counts())
print(data_proms['chemo_change'].value_counts())
print(data_C30['admissions'].value_counts())
print(data_C30['triage'].value_counts())
print(data_C30['chemo_change'].value_counts())
print(data_5D['admissions'].value_counts())
print(data_5D['triage'].value_counts())
print(data_5D['chemo_change'].value_counts())
print(data_fact['admissions'].value_counts())
print(data_fact['triage'].value_counts())
print(data_fact['chemo_change'].value_counts())


# In[ ]:


data_all_majority_admissions = data_all[data_all.admissions==0]
data_all_minority_admissions = data_all[data_all.admissions==1]
data_all_minority_upsampled_admissions = resample(data_all_minority_admissions, 
                                 replace=True,     
                                 n_samples=277,    
                                 random_state=123) 
data_all_upsampled_admissions = pd.concat([data_all_majority_admissions, data_all_minority_upsampled_admissions])


# In[ ]:


data_all_majority_triage = data_all[data_all.triage==1]
data_all_minority_triage = data_all[data_all.triage==0]
data_all_minority_upsampled_triage = resample(data_all_minority_triage, 
                                 replace=True,     
                                 n_samples=240,    
                                 random_state=123) 
data_all_upsampled_triage = pd.concat([data_all_majority_triage, data_all_minority_upsampled_triage])


# In[ ]:


data_all_majority_chemo_change = data_all[data_all.chemo_change==1]
data_all_minority_chemo_change = data_all[data_all.chemo_change==0]
data_all_minority_upsampled_chemo_change = resample(data_all_minority_chemo_change, 
                                 replace=True,     
                                 n_samples=279,    
                                 random_state=123) 
data_all_upsampled_chemo_change = pd.concat([data_all_majority_chemo_change, data_all_minority_upsampled_chemo_change])


# In[ ]:


data_clinical_majority_admissions = data_clinical[data_clinical.admissions==0]
data_clinical_minority_admissions = data_clinical[data_clinical.admissions==1]
data_clinical_minority_upsampled_admissions = resample(data_clinical_minority_admissions, 
                                 replace=True,     
                                 n_samples=338,    
                                 random_state=123) 
data_clinical_upsampled_admissions = pd.concat([data_clinical_majority_admissions, data_clinical_minority_upsampled_admissions])


# In[ ]:


data_clinical_majority_triage = data_clinical[data_clinical.triage==1]
data_clinical_minority_triage = data_clinical[data_clinical.triage==0]
data_clinical_minority_upsampled_triage = resample(data_clinical_minority_triage, 
                                 replace=True,     
                                 n_samples=294,    
                                 random_state=123) 
data_clinical_upsampled_triage = pd.concat([data_clinical_majority_triage, data_clinical_minority_upsampled_triage])


# In[ ]:


data_clinical_majority_chemo_change = data_clinical[data_clinical.chemo_change==1]
data_clinical_minority_chemo_change = data_clinical[data_clinical.chemo_change==0]
data_clinical_minority_upsampled_chemo_change = resample(data_clinical_minority_chemo_change, 
                                 replace=True,     
                                 n_samples=333,    
                                 random_state=123) 
data_clinical_upsampled_chemo_change = pd.concat([data_clinical_majority_chemo_change, data_clinical_minority_upsampled_chemo_change])


# In[ ]:


data_proms_majority_admissions = data_proms[data_proms.admissions==0]
data_proms_minority_admissions = data_proms[data_proms.admissions==1]
data_proms_minority_upsampled_admissions = resample(data_proms_minority_admissions, 
                                 replace=True,     
                                 n_samples=277,    
                                 random_state=123) 
data_proms_upsampled_admissions = pd.concat([data_proms_majority_admissions, data_proms_minority_upsampled_admissions])


# In[ ]:


data_proms_majority_triage = data_proms[data_proms.triage==1]
data_proms_minority_triage = data_proms[data_proms.triage==0]
data_proms_minority_upsampled_triage = resample(data_proms_minority_triage, 
                                 replace=True,     
                                 n_samples=240,    
                                 random_state=123) 
data_proms_upsampled_triage = pd.concat([data_proms_majority_triage, data_proms_minority_upsampled_triage])


# In[ ]:


data_proms_majority_chemo_change = data_proms[data_proms.chemo_change==1]
data_proms_minority_chemo_change = data_proms[data_proms.chemo_change==0]
data_proms_minority_upsampled_chemo_change = resample(data_proms_minority_chemo_change, 
                                 replace=True,     
                                 n_samples=279,    
                                 random_state=123) 
data_proms_upsampled_chemo_change = pd.concat([data_proms_majority_chemo_change, data_proms_minority_upsampled_chemo_change])


# In[ ]:


data_C30_majority_admissions = data_C30[data_C30.admissions==0]
data_C30_minority_admissions = data_C30[data_C30.admissions==1]
data_C30_minority_upsampled_admissions = resample(data_C30_minority_admissions, 
                                 replace=True,     
                                 n_samples=277,    
                                 random_state=123) 
data_C30_upsampled_admissions = pd.concat([data_C30_majority_admissions, data_C30_minority_upsampled_admissions])


# In[ ]:


data_C30_majority_triage = data_C30[data_C30.triage==1]
data_C30_minority_triage = data_C30[data_C30.triage==0]
data_C30_minority_upsampled_triage = resample(data_C30_minority_triage, 
                                 replace=True,     
                                 n_samples=240,    
                                 random_state=123) 
data_C30_upsampled_triage = pd.concat([data_C30_majority_triage, data_C30_minority_upsampled_triage])


# In[ ]:


data_C30_majority_chemo_change = data_C30[data_C30.chemo_change==1]
data_C30_minority_chemo_change = data_C30[data_C30.chemo_change==0]
data_C30_minority_upsampled_chemo_change = resample(data_C30_minority_chemo_change, 
                                 replace=True,     
                                 n_samples=279,    
                                 random_state=123) 
data_C30_upsampled_chemo_change = pd.concat([data_C30_majority_chemo_change, data_C30_minority_upsampled_chemo_change])


# In[ ]:


data_5D_majority_admissions = data_5D[data_5D.admissions==0]
data_5D_minority_admissions = data_5D[data_5D.admissions==1]
data_5D_minority_upsampled_admissions = resample(data_5D_minority_admissions, 
                                 replace=True,     
                                 n_samples=338,    
                                 random_state=123) 
data_5D_upsampled_admissions = pd.concat([data_5D_majority_admissions, data_5D_minority_upsampled_admissions])


# In[ ]:


data_5D_majority_triage = data_5D[data_5D.triage==1]
data_5D_minority_triage = data_5D[data_5D.triage==0]
data_5D_minority_upsampled_triage = resample(data_5D_minority_triage, 
                                 replace=True,     
                                 n_samples=294,    
                                 random_state=123) 
data_5D_upsampled_triage = pd.concat([data_5D_majority_triage, data_5D_minority_upsampled_triage])


# In[ ]:


data_5D_majority_chemo_change = data_5D[data_5D.chemo_change==1]
data_5D_minority_chemo_change = data_5D[data_5D.chemo_change==0]
data_5D_minority_upsampled_chemo_change = resample(data_5D_minority_chemo_change, 
                                 replace=True,     
                                 n_samples=333,    
                                 random_state=123) 
data_5D_upsampled_chemo_change = pd.concat([data_5D_majority_chemo_change, data_5D_minority_upsampled_chemo_change])


# In[ ]:


data_fact_majority_admissions = data_fact[data_fact.admissions==0]
data_fact_minority_admissions = data_fact[data_fact.admissions==1]
data_fact_minority_upsampled_admissions = resample(data_fact_minority_admissions, 
                                 replace=True,     
                                 n_samples=338,    
                                 random_state=123) 
data_fact_upsampled_admissions = pd.concat([data_fact_majority_admissions, data_fact_minority_upsampled_admissions])


# In[ ]:


data_fact_majority_triage = data_fact[data_fact.triage==1]
data_fact_minority_triage = data_fact[data_fact.triage==0]
data_fact_minority_upsampled_triage = resample(data_fact_minority_triage, 
                                 replace=True,     
                                 n_samples=294,    
                                 random_state=123) 
data_fact_upsampled_triage = pd.concat([data_fact_majority_triage, data_fact_minority_upsampled_triage])


# In[ ]:


data_fact_majority_chemo_change = data_fact[data_fact.chemo_change==1]
data_fact_minority_chemo_change = data_fact[data_fact.chemo_change==0]
data_fact_minority_upsampled_chemo_change = resample(data_fact_minority_chemo_change, 
                                 replace=True,     
                                 n_samples=333,    
                                 random_state=123) 
data_fact_upsampled_chemo_change = pd.concat([data_fact_majority_chemo_change, data_fact_minority_upsampled_chemo_change])


# In[ ]:


print(data_all_upsampled_admissions['admissions'].value_counts())
print(data_all_upsampled_triage['triage'].value_counts())
print(data_all_upsampled_chemo_change['chemo_change'].value_counts())
print(data_clinical_upsampled_admissions['admissions'].value_counts())
print(data_clinical_upsampled_triage['triage'].value_counts())
print(data_clinical_upsampled_chemo_change['chemo_change'].value_counts())
print(data_proms_upsampled_admissions['admissions'].value_counts())
print(data_proms_upsampled_triage['triage'].value_counts())
print(data_proms_upsampled_chemo_change['chemo_change'].value_counts())
print(data_C30_upsampled_admissions['admissions'].value_counts())
print(data_C30_upsampled_triage['triage'].value_counts())
print(data_C30_upsampled_chemo_change['chemo_change'].value_counts())
print(data_5D_upsampled_admissions['admissions'].value_counts())
print(data_5D_upsampled_triage['triage'].value_counts())
print(data_5D_upsampled_chemo_change['chemo_change'].value_counts())
print(data_fact_upsampled_admissions['admissions'].value_counts())
print(data_fact_upsampled_triage['triage'].value_counts())
print(data_fact_upsampled_chemo_change['chemo_change'].value_counts())


# In[ ]:


y_a_all = data_all.admissions
y_t_all = data_all.triage
y_c_all = data_all.chemo_change

y_a_clinical = data_clinical.admissions
y_t_clinical = data_clinical.triage
y_c_clinical = data_clinical.chemo_change

y_a_proms = data_proms.admissions
y_t_proms = data_proms.triage
y_c_proms = data_proms.chemo_change

y_a_C30 = data_C30.admissions
y_t_C30 = data_C30.triage
y_c_C30 = data_C30.chemo_change

y_a_5D = data_5D.admissions
y_t_5D = data_5D.triage
y_c_5D = data_5D.chemo_change

y_a_fact = data_fact.admissions
y_t_fact = data_fact.triage
y_c_fact = data_fact.chemo_change


# In[ ]:


y_a_all_up = data_all_upsampled_admissions.admissions
y_t_all_up = data_all_upsampled_triage.triage
y_c_all_up = data_all_upsampled_chemo_change.chemo_change

y_a_clinical_up = data_clinical_upsampled_admissions.admissions
y_t_clinical_up = data_clinical_upsampled_triage.triage
y_c_clinical_up = data_clinical_upsampled_chemo_change.chemo_change

y_a_proms_up = data_proms_upsampled_admissions.admissions
y_t_proms_up = data_proms_upsampled_triage.triage
y_c_proms_up = data_proms_upsampled_chemo_change.chemo_change

y_a_C30_up= data_C30_upsampled_admissions.admissions
y_t_C30_up = data_C30_upsampled_triage.triage
y_c_C30_up = data_C30_upsampled_chemo_change.chemo_change

y_a_5D_up = data_5D_upsampled_admissions.admissions
y_t_5D_up = data_5D_upsampled_triage.triage
y_c_5D_up = data_5D_upsampled_chemo_change.chemo_change

y_a_fact_up = data_fact_upsampled_admissions.admissions
y_t_fact_up = data_fact_upsampled_triage.triage
y_c_fact_up = data_fact_upsampled_chemo_change.chemo_change


# In[ ]:


x_all = data_all.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_clinical = data_clinical.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_proms = data_proms.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_C30 = data_C30.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_5D = data_5D.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_fact = data_fact.drop(['admissions', 'triage', 'chemo_change'], axis=1)


# In[ ]:


x_all_admissions = data_all_upsampled_admissions.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_all_triage = data_all_upsampled_triage.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_all_chemo = data_all_upsampled_chemo_change.drop(['admissions', 'triage', 'chemo_change'], axis=1)

x_clinical_admissions = data_clinical_upsampled_admissions.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_clinical_triage = data_clinical_upsampled_triage.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_clinical_chemo = data_clinical_upsampled_chemo_change.drop(['admissions', 'triage', 'chemo_change'], axis=1)

x_proms_admissions = data_proms_upsampled_admissions.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_proms_triage = data_proms_upsampled_triage.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_proms_chemo = data_proms_upsampled_chemo_change.drop(['admissions', 'triage', 'chemo_change'], axis=1)

x_C30_admissions = data_C30_upsampled_admissions.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_C30_triage = data_C30_upsampled_triage.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_C30_chemo = data_C30_upsampled_chemo_change.drop(['admissions', 'triage', 'chemo_change'], axis=1)

x_5D_admissions = data_5D_upsampled_admissions.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_5D_triage = data_5D_upsampled_triage.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_5D_chemo= data_5D_upsampled_chemo_change.drop(['admissions', 'triage', 'chemo_change'], axis=1)

x_fact_admissions = data_fact_upsampled_admissions.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_fact_triage = data_fact_upsampled_triage.drop(['admissions', 'triage', 'chemo_change'], axis=1)
x_fact_chemo= data_fact_upsampled_chemo_change.drop(['admissions', 'triage', 'chemo_change'], axis=1)


# In[ ]:


def get_scores (X, Y):
    models = [LogisticRegression(solver= 'liblinear'),DecisionTreeClassifier(), AdaBoostClassifier(), RandomForestClassifier(), NuSVC() ]
    names = ['LOGISTIC REGRESSION', "DECISION TREE",  'Adaptive Boosting', 'RANDOM FOREST', 'SVM']
    scoring_methods=['accuracy','precision','recall','f1','roc_auc']
    for model, name in zip(models, names):
        print (name)
        scores = cross_validate(model, X, Y, cv=10,
                                scoring= scoring_methods,
                                return_train_score=False)
        print(np.mean(scores['test_accuracy']))
        print(np.mean(scores['test_precision']))
        print(np.mean(scores['test_recall']))
        print(np.mean(scores['test_f1']))
        print(np.mean(scores['test_roc_auc']))
    


# In[ ]:


get_scores (x_all, y_a_all)


# In[ ]:


get_scores (x_all, y_t_all)


# In[ ]:


get_scores (x_all, y_c_all)


# In[ ]:


get_scores (x_clinical, y_a_clinical)


# In[ ]:


get_scores (x_clinical, y_t_clinical)


# In[ ]:


get_scores (x_clinical, y_c_clinical)


# In[ ]:


get_scores (x_proms, y_a_proms)


# In[ ]:


get_scores (x_proms, y_t_proms)


# In[ ]:


get_scores (x_proms, y_c_proms)


# In[ ]:


get_scores (x_C30, y_a_C30)


# In[ ]:


get_scores (x_C30, y_t_C30)


# In[ ]:


get_scores (x_C30, y_c_C30)


# In[ ]:


get_scores (x_5D, y_a_5D)


# In[ ]:


get_scores (x_5D, y_t_5D)


# In[ ]:


get_scores (x_5D, y_c_5D)


# In[ ]:


get_scores (x_fact, y_a_fact)


# In[ ]:


get_scores (x_fact, y_t_fact)


# In[ ]:


get_scores (x_fact, y_c_fact)


# In[ ]:


get_scores (x_all_admissions, y_a_all_up)


# In[ ]:


get_scores (x_all_triage, y_t_all_up)


# In[ ]:


get_scores (x_all_chemo, y_c_all_up)


# In[ ]:


get_scores (x_clinical_admissions, y_a_clinical_up)


# In[ ]:


get_scores (x_clinical_triage, y_t_clinical_up)


# In[ ]:


get_scores (x_clinical_chemo, y_c_clinical_up)


# In[ ]:


get_scores (x_C30_admissions, y_a_C30_up)


# In[ ]:


get_scores (x_C30_triage, y_t_C30_up)


# In[ ]:


get_scores (x_C30_chemo, y_c_C30_up)


# In[ ]:


get_scores (x_proms_admissions, y_a_proms_up)


# In[ ]:


get_scores (x_proms_triage, y_t_proms_up)


# In[ ]:


get_scores (x_proms_chemo, y_c_proms_up)


# In[ ]:


get_scores (x_5D_admissions, y_a_5D_up)


# In[ ]:


get_scores (x_5D_triage, y_t_5D_up)


# In[ ]:


get_scores (x_5D_chemo, y_c_5D_up)


# In[ ]:


get_scores (x_fact_admissions, y_a_fact_up)


# In[ ]:


get_scores (x_fact_triage, y_t_fact_up)


# In[ ]:


get_scores (x_fact_chemo, y_c_fact_up)


# In[ ]:


results = pd.read_csv('results.csv')
results


# In[ ]:


import statistics
results.mean()


# In[ ]:


from scipy import stats
import matplotlib.pyplot as plt


# In[ ]:


stats.probplot(results["AUCupc"], dist="norm", plot=plt)


# In[ ]:


from scipy.stats import mannwhitneyu


# In[ ]:


print(mannwhitneyu(results['accuracy'], results['accuracyup']))
print(mannwhitneyu(results['precision'], results['precisionup']))
print(mannwhitneyu(results['recall'], results['recallup']))
print(mannwhitneyu(results['f1 score'], results['f1 scoreup']))
print(mannwhitneyu(results['AUC'], results['AUCup']))


# In[ ]:


print(mannwhitneyu(results['accuracyt'], results['accuracyupt']))
print(mannwhitneyu(results['precisiont'], results['precisionupt']))
print(mannwhitneyu(results['recallt'], results['recallupt']))
print(mannwhitneyu(results['f1 scoret'], results['f1 scoreupt']))
print(mannwhitneyu(results['AUCt'], results['AUCupt']))


# In[ ]:


print(mannwhitneyu(results['accuracyc'], results['accuracyupc']))
print(mannwhitneyu(results['precisionc'], results['precisionupc']))
print(mannwhitneyu(results['recallc'], results['recallupc']))
print(mannwhitneyu(results['f1 scorec'], results['f1 scoreupc']))
print(mannwhitneyu(results['AUCc'], results['AUCupc']))


# In[ ]:




