#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering CW
# 

# ### Import required packages

# In[75]:


# to handle datasets
import pandas as pd
import numpy as np

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)
# suppress some warning
pd.options.mode.chained_assignment = None  # default='warn'


# ### Load the data and Plot a Histogram of the SalePrice column

# In[76]:


# load dataset
data = pd.read_csv('house-price-data.csv')
df=data
data.hist(column='SalePrice',  bins = 250)


# ### 1- The SalePrice column is not normally distributed (i.e. not Gaussian), prove this by running a statistical test and obtaining and interpreting the p-value (you can use if else to check the p-value and interpret it). (5 Marks)

# In[77]:


## Perform Shapiro-Wilk test for normaility of the Sale Price data
from scipy.stats import shapiro
shapiro(data['SalePrice'])


# from the Shaprio-Wilk test we can see that the p value is 3.206 e-33. The p-value is less than 0.05, we reject the null hypothesis of the Shapiro-Wilk test. This means we can say that the sale price data does not come from a normal distribution.

# ### Split data into Train and Test sub-datasets
# ### Do not change this code

# In[78]:


from sklearn.model_selection import train_test_split
### Split data into train and test sets 
y = data['SalePrice']
X = data.drop(columns=['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=10)

X_train.shape, X_test.shape


# ### 2. Create a list of all categorical variables (by checking their type in the original dataset). (2 marks)
# 

# In[79]:


# let's identify the categorical variables through info()
data.info()

# we will create a list of categorical variable by selecting object types
cat_vars = data.select_dtypes("object").columns
print(cat_vars)

# number of categorical variables by calling it through list in len function to avoid the total counts
len(cat_vars)
print(X_train.shape)


# ### 3. Using the training set (X_train) Create a list of all categorical variables that contain missing data and print percentage of missing values per variable. (3 marks)

# In[80]:


# make a list of the categorical variables that contain missing values
cat_vars_miss = [var for var in X_train.columns if X_train[var].isnull().mean()>0 and X_train[var].dtypes ==('object')]
cat_vars_miss

# First calculate the percentage of all missing values in the data
all_data_na = (X_train.isnull().sum() / len(X_train)) * 100

# select the only categorical variable from the data by using drop function for the numerical data
cat_missing_data =all_data_na.drop(all_data_na[ X.select_dtypes("number").columns].index).sort_values(ascending=False)[:30]

#used iloc function to get only categorical varibale with missing value
cat_missing_per= cat_missing_data.iloc[:11]
cat_missing_per


# ### 4. Using the result of the previous step: For categorical variables with more than 10% of data missing, replace missing data with the word 'Missing', in other variables replace the missing data with the most frequent category in  the training set (Apply the replacement to X_train and X_test and make sure it is based on the results you have obtained from the training set). (5 marks)

# In[81]:


# variables to impute with the string missing

with_string_missing = [var for var in X_train.columns if X_train[var].isnull().mean() >0.1 and X_train[var].dtypes ==('object')]

# variables to impute with the most frequent category

with_frequent_category = [var for var in cat_vars_miss if var not in with_string_missing]


# In[82]:


print(with_string_missing)
print(with_frequent_category)
X_test[with_string_missing].isnull().sum() # checking how much null values of with_string_missing holds


# In[83]:


# replace missing values in X_train and X_test with new label: "Missing"
#for var in with_string_missing: # use for loop in with_string_missing list to fillna 
X_train[with_string_missing]= X_train[with_string_missing].fillna('Missing')
X_test[with_string_missing] = X_test[with_string_missing].fillna('Missing')
X_test[with_string_missing].isnull().sum() # checking to see if missing values has been replaced by 'Missing'


# In[84]:


# replace missing values in X_train and X_test with the mode of each variable
for var in with_frequent_category:
    frequent_category = X_train[var].value_counts().index[0] # use loop for count the number of category in each variable and select the most frequent category by using index[0]
    X_train[var]= X_train[var].fillna(frequent_category)
    X_test[var]= X_test[var].fillna(frequent_category)
  


# In[85]:


# check that we have no missing information in the engineered variables
X_train[cat_vars_miss].isnull().sum()


# In[86]:


#Checking if there is any missing information in test data set as well
X_test[cat_vars_miss].isnull().sum()


# ### 5. Create a list of all numerical variables (do not include SalePrice). (2 marks)
# Correct solution says there are 35 of them.

# In[87]:


# now let's identify the numerical variables
num_vars = X.select_dtypes('number').columns # use X data to exclude SalePrice column
num_vars

# number of numerical variables
len(num_vars) 


# ### 6. Create a list of all numerical variables that contain missing data and print out the percentage of missing values per variable (use the training data). (3 marks)

# In[88]:


# initialise an empty list to store the variables and percentage of missing values

num_missing_data = []

# use for loop to iterate through the numerical variables
for var in num_vars:
    percent_missing = X_train[var].isnull().mean()*100 # percentage of missing value
    if percent_missing >0: # use if condition to select missing numerical data and add variable and percentage to list
        num_missing_data.append((var,percent_missing))
        
# print percentage of missing values per variable
print(num_missing_data)


# ### 7. Using the result of the previous step: For numerical variables with less than 15% of data missing, replace missing data with the mean of the variable, in other variables replace the missing data with the median of the variable in the training set (Apply the replacement to X_train and X_test and make sure it is based on the results you have obtained from the training set).     (5 marks)

# In[89]:


#Use for  loop in the num_var to iterate the columns
for cols in num_vars:
    
    percent_missing = X_train[cols].isnull().mean()*100 # calculate the percenatge of missing values
    
    if percent_missing <15: # if condition again to choose numerical variables with less than 15% missing data
        
        mean = X_train[cols].mean() #calculate the mean
        
        X_train[cols].fillna(mean, inplace= True) # replace the missing data with mean in both  data sets(X-test and X-train)
        X_test[cols].fillna(mean, inplace= True)
        
    else:   # efficient to use else condition condition to replace the remaining missing values in the numerical variable
                                             # with median
        
        median = X_train[cols].median()
        X_train[cols].fillna(median,inplace =True)
        X_test[cols].fillna(median,inplace =True)


# In[90]:


# check that we have no more missing values in the engineered variables
X_train[num_vars].isnull().sum()
X_test[num_vars].isnull().sum()


# ### 8. In the train and test sets, replace the values of variables 'YearBuilt', 'YearRemodAdd' and 'GarageYrBlt' with the time elapsed between them and the year in which the house was sold 'YrSold' and then drop the  'YrSold' column (5 marks)

# In[91]:


# First calculate the time elapsed between variables and the Year sold by simply subtracting it. It will then replace the values
# of the variable with the time elapsed between them.
X_train['YearBuilt']      =    X_train ['YrSold'] - X_train ['YearBuilt']
X_train['YearRemodAdd']   =    X_train ['YrSold'] - X_train['YearRemodAdd']
X_train['GarageYrBlt']    =    X_train ['YrSold'] - X_train['GarageYrBlt'] 
X_test ['YearBuilt']      =    X_test ['YrSold'] - X_test ['YearBuilt']
X_test['YearRemodAdd']    =    X_test ['YrSold'] - X_test['YearRemodAdd']
X_test ['GarageYrBlt']    =    X_test ['YrSold'] - X_test['GarageYrBlt'] 


# In[92]:


# now we drop YrSold
X_train.drop(columns =['YrSold'], inplace=True)
X_test.drop(columns=['YrSold'], inplace=True)


# ### 9. Apply mappings to categorical variables that have an order (5 marks)
# 
# Some of the categorical variables have values with an assigned order (in total there should be 14 of them), related to quality (For more information, check the data description file). This means, we can replace categories by numbers to determine quality. For example, values in the 'BsmtExposure' can be mapped as follows: 'No' can be mapped to 1, 'Mn' can be mapped to 2, 'Av' can be mapped to 3 and 'Gd' can be mapped to 4. 
# 
# One way of doing this is to manually create mappings similar to the example given. Each mapping can be saved as a Python dictionary and used to perform the actual mapping to transform the described variables from categorical to numerical.
# 
# To Make it easier for you, here are groups of variables that have the same mappings (Hint: you can map both categories 'Missing' and 'NA' to 0): 
# 
# - The following variable groups have the same mapping: 
#     - ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu','GarageQual', 'GarageCond']
#     - ['BsmtFinType1', 'BsmtFinType2']
# 
# - Each of the following variables has its own mapping: 'BsmtExposure', 'GarageFinish', 'Fence'

# In[93]:


# Create a list of categorical variable with same qualities
qual_vars= ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond']

# Iterate through the list of variables
for var in qual_vars:
    #Create a mapping to change categorical value to numerical values
    qual_mappings = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}
    
    # map the values of the variable through mapping
    X_train[var] = X_train[var].map(qual_mappings)
    # replacing missing values with 'Missing'
    X_train[var] = X_train [var].fillna('Missing')

# same method applied for X_test data set

for var in qual_vars:
    qual_mappings = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'Missing': 0, 'NA': 0}
    X_test[var] = X_test[var].map(qual_mappings)
    X_test[var]= X_test[var].fillna('Missing')
    
X_test[qual_vars].isnull().sum() # checking if mapping has been performed correctly


# In[94]:



var = 'BsmtExposure'

exposure_mappings = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

## Apply the mapping directly to the train and test sets
X_train[var] = X_train[var].map (exposure_mappings)
X_train[var] = X_train[var].fillna('Missing')
X_test[var] = X_test[var].map(exposure_mappings)
X_test[var] = X_test[var].fillna('Missing')


X_test[var].isnull().sum()


# In[95]:


finish_vars = ['BsmtFinType1', 'BsmtFinType2']

for var in finish_vars:
    finish_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    X_train[var] = X_train[var].map(finish_mappings)
    X_train[var] = X_train[var].fillna ('Missing')

# same method applied for X_test data set

for var in finish_vars:
    finish_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    X_test[var] = X_test[var].map(finish_mappings)
    X_test[var] = X_test[var].fillna ('Missing')
    

X_train[finish_vars].isnull().sum()


# In[96]:


garage_mappings = {'Missing': 0, 'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

var = 'GarageFinish'

X_train[var] = X_train[var].map(garage_mappings)
X_train[var]= X_train[var].fillna('Missing')

X_test['GarageFinish'] = X_test['GarageFinish'].map(garage_mappings)
X_test[var] = X_test[var].fillna('Missing')

X_test[var].isnull().sum()


# In[97]:


fence_mappings = {'Missing': 0, 'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}

# use the variable name directly rather than assigning other name
X_train['Fence'] = X_train['Fence'].map(fence_mappings)
X_train['Fence'] = X_train['Fence'].fillna('Missing')
X_test['Fence'] = X_test['Fence'].map(fence_mappings)
X_test['Fence'] = X_test['Fence'].fillna('Missing')

X_test['Fence'].isnull().sum()
X_train['Fence'].isnull().sum()


# In[98]:


# check absence of na in the train set
[var for var in X_train.columns if X_train[var].isnull().sum() > 0]


# In[99]:


X_train.head()
X_test.shape


# ### 10. Replace Rare Labels with 'Rare' (5 marks)
# 
# For the remaining five categorical variables (the variables that you did not apply value mappings to, they should be five variables), you will need to group those categories that are present in less than 1% of the observations in the training set. That is, all values of categorical variables that are shared by less than 1% of houses in the training set will be replaced by the string "Rare" in both the training and test set. 
# 
# - Find rare labels in the remaining categorical variables and replace them with the category 'Rare'.
# - Rare labels are those categories that only appear in a small percentage of the observations (in our case in < 1%).
# - If you look at unique values in a categorical variable in the training set and count how many times each of the unique values appear in the variable, you can compute the percentage of each unique value by dividing its count by the total number of observations.
# - Remember to make the computions using the training set and replacement in both training and test sets.

# In[100]:


# capture all quality variables

qual_vars  = qual_vars + finish_vars + ['BsmtExposure','GarageFinish','Fence']

# capture the remaining categorical variables

# (those that we did not re-map)

cat_others = [var for var in cat_vars if var not in qual_vars]
len(cat_others)
print(cat_others) # print cat_others list t
X_train.shape


# In[101]:


# Compute the threshold for identifying rare labels
threshold = 0.01 * len(X_train)

# Iterate over all cat_others variables
for col in cat_others:
    # Compute the counts of each unique value in the column
    value_counts = X_train[col].value_counts()
    
    # Identify the rare labels
    rare_labels = value_counts[value_counts < threshold]
    
    # Replace the rare labels with the 'Rare' string in both the training and test sets
    X_train[col] = X_train[col].replace({label: 'Rare' for label in rare_labels.index})
    X_test[col] = X_test[col].replace({label: 'Rare' for label in rare_labels.index})


X_test.shape
X_train.shape
# check if 'Rare' label 
#Get the unique values in the column
unique_values = X_train[col].unique()

# Print the unique values to check if rare label has been replaced by 'Rare'
print(unique_values)


# ### 11. One hot encoding of categorical variables (5 marks)
# Perform one hot encoding to transform the previous five categorical variables into binary variables. Make sure you do it correctly for both the training and testing sets. After this, remember to drop the original five categorical variables (the ones with the strings) from the training and test after the encoding.

# In[102]:



for var in cat_others:             #One hot encoding for each categorical variable
                                   # Get one hot encoding of the variable
        one_hot = pd.get_dummies(X_train[var], prefix=var)
        # Drop the original variable from the dataframe
        X_train = X_train.drop(var, axis=1)
        # Concatenate the one hot encoded variables to the dataframe
        X_train = pd.concat([X_train, one_hot], axis=1)
for var in cat_others:
    one_hot = pd.get_dummies(X_test[var], prefix=var)
    X_test = X_test.drop(var,axis=1)
    X_test = pd.concat([X_test, one_hot], axis=1)


# In[103]:


# check absence of na in the train set
[var for var in X_train.columns if X_train[var].isnull().sum() > 0]


# In[104]:


# check absence of na in the test set
[var for var in X_test.columns if X_test[var].isnull().sum() > 0]


# ### 12. Feature Scaling (5 marks)
# Now we know all variables in our two datasets (i.e. the training and test sets) are numerical, the final step in this exercise is to apply scaling by making sure the minimum value in each variable is 0 and the maximum value is 1. For this step, you can use MinMaxScaler() from sci-kit learn. Make sure you apply it correctly by transforming the test set based on the training set. 

# In[105]:


#import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

#Initialise scaler
scaler= MinMaxScaler()

#Apply scaler on X-test data set
scaler.fit(X_train)

#Transfrom train and test data sets
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


print(X_train.shape)
print(X_test.shape)


# In[106]:



print(X_train.mean().mean())
print(X_test.mean().mean())


# 
# # Well done!

# In[ ]:




