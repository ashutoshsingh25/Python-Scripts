#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv("c:/Users/imart/Desktop/Book1.csv") #specify path to the input file
#take only required column names into a new Dataframe Data
Data=data[['FK_GLCAT_MCAT_ID','ETO_OFR_TITLE','GLUSR_USR_CITY','FK_IM_SPEC_MASTER_DESC','FK_IM_SPEC_OPTIONS_DESC','UNIQUE_SOLD','RETAIL_FLAG','EMAIL_FLAG','PRIME_MCAT_NAME','PORPOSE','RETAIL_BL_NI','SECONDARY_MCAT','FREQUENCY']].copy()
a=Data.sort_values(by=['PRIME_MCAT_NAME'])
#creating replace map for feature PORPOSE
replace_map = {'PORPOSE': {'0': 0, 'personal': 'personal_use', 'business_use': 'business', 'reselling': 'reselling_use'}}
Data.replace(replace_map, inplace=True)
Data = pd.get_dummies(Data, columns=['PORPOSE'], prefix = ['PORPOSE'])
val=Data['FK_GLCAT_MCAT_ID'].unique().tolist()
print (val)
# splitting the dataframe into different dataframes as per their MCAT ids.
d = dict(tuple(Data.groupby('FK_GLCAT_MCAT_ID')))

#df.to_csv('c:/Users/imart/Desktop/agriculturalmachinery.csv', index='false')


# In[2]:


# Function call to split the secondary MCAT names as different entries 
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
for i in range(0,len(val))  :  
    d[val[i]]=explode(d[val[i]].assign(SECONDARY_MCAT=d[val[i]].SECONDARY_MCAT.str.split('|')), 'SECONDARY_MCAT')
    #d[val[i]].to_csv('c:/Users/imart/Desktop/1.csv', index='false')


# In[3]:


for i in range(0,len(val))  :  
    print (d[val[i]])


# In[4]:


# For each MCAT dataframe we features to work on
for i in range(0,len(val)):
    #combining one ISQ_spec with all its possible options
    d[val[i]]['ISQ_OPTION'] = d[val[i]]['FK_IM_SPEC_MASTER_DESC'] + " | "+d[val[i]]['FK_IM_SPEC_OPTIONS_DESC'] 
    # generating one hot vectors for all the features
    d[val[i]] = pd.get_dummies(d[val[i]], columns=['PRIME_MCAT_NAME'], prefix = ['PRIME_MCAT'])
    d[val[i]] = pd.get_dummies(d[val[i]], columns=['GLUSR_USR_CITY'], prefix = ['CITY'])
    d[val[i]] = pd.get_dummies(d[val[i]], columns=['ISQ_OPTION'], prefix = ['ISQ_OPTION'])
    d[val[i]] = pd.get_dummies(d[val[i]], columns=['SECONDARY_MCAT'], prefix = ['SEC_MCAT'])
    d[val[i]] = pd.get_dummies(d[val[i]], columns=['FREQUENCY'], prefix = ['FREQUENCY'])


# In[5]:


#vectorizing the title name
title1=[]
for i in range(0,len(val)): 
    a= d[val[i]]['ETO_OFR_TITLE']
#print (len(a))
    count= len(d[val[i]]['ETO_OFR_TITLE'])
    print (count)
    list1= (list(a))
    print (list1)
    #print (len(list1))
    title_vect=[]
    
    for q in range(0, len(a)):
    
        print (list1[q],list1.count(list1[q]))
        title_vect.append((float(list1.count(list1[q])/count)))
    print (title_vect)
    title1.extend(title_vect)
    d[val[i]]['TITLE_VECTOR']=pd.DataFrame(title_vect)


# In[6]:


#importing the required packages
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
#storing all unique mcat names
li=Data['PRIME_MCAT_NAME'].unique().tolist()
print (li)


# In[19]:


#to solve the problem of class imbalance smote is being used
from imblearn.over_sampling import SMOTE

for i in range(0,len(val)):
    
    print (val[i])
    #X contains all the required features and Y will have the label/target value
    X = d[val[i]].drop(['FK_GLCAT_MCAT_ID','ETO_OFR_TITLE','FK_IM_SPEC_MASTER_DESC','FK_IM_SPEC_OPTIONS_DESC','UNIQUE_SOLD'], axis=1)  
    y = d[val[i]]['UNIQUE_SOLD']  
      
    #Applying SMOTE 
    
    smote = SMOTE(ratio='minority')
    X_sm, y_sm = smote.fit_sample(X, y)
    #splitting the data into training and testing dataset (using 60:40 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40,random_state=1,shuffle='true') 
    from sklearn.svm import SVC  
    #creating SVM with linear kernel (tuning the C and gamma parameter)
    model = SVC(kernel='linear',C=1000,gamma='scale',class_weight='balanced')
    #fitting the model on our test set
    model.fit(X_train, y_train) 
    #running the model on the test set
    predicted=model.predict(X_test)
    
    #storing the weights of the fetaures
    coeff=model.coef_
    #print (coeff)
    #calculating the accuracy of the model
    acc_score=accuracy_score(y_test,predicted)
    print ("Accuracy:", acc_score)
    for i in range(0,len(coeff)):
        a=coeff[i]
    
    #list of features  
    feat=list(X)
    #print (feat)
    # creating a dataframe with features and their corresponding weights
    df2 = pd.DataFrame({'Features': feat, 'Weights': a})
    
    print (df2)
    #saving results to a csv for each MCAT
    df2.to_csv('C:\\Users\\imart\\Desktop\\mcat_details\\weights_'+li[i]+'.csv', index='false')
    


# In[ ]:





# In[11]:


# making csv files for different MCAT dataframes that will serve as input to the model
for i in range(0,len(val)):
    d[val[i]].to_csv('C:\\Users\\imart\\Desktop\\mcat_details\\data\\'+li[i]+'.csv', index='false')


# In[ ]:





# In[ ]:




