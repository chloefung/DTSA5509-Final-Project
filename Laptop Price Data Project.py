#!/usr/bin/env python
# coding: utf-8

# # Laptop Price Data Project
# 
# The aim of this study is to utilize mechine learning techniques to uncover and analyze the most relevant factors affecting the laptop price.

# ## Part 1: Load the Data
# 
# This study is using the Laptop Price Dataset available from Kaggle, (URL: https://www.kaggle.com/datasets/gyanprakashkushwaha/laptop-price-prediction-cleaned-dataset). 

# In[1]:


import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import statsmodels.api as sm


# In[2]:


df = pd.read_csv('data/laptop_data.csv')
df.head()


# ## Part 2: Check the Data Types

# In[3]:


df.info()


# Based on the information of the dataset, there is no error found in data type.

# ## Part 3: Data Cleaning
# 
# We can drop the features that are unnecessary. `Company`, `TypeName`, `Cpu_brand`, `Gpu_brand` and `Os` are object and not a meaningful feature so we can remove it and replace the df.

# In[4]:


df = df.drop(columns=['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand','Os'])


# Reorder the columns so the `Price` (the target variable) is the first column.

# In[5]:


df = df.iloc[:,[2,0,1,3,4,5,6,7]]


# In[6]:


df.head()


# ## Part 4: Correlation Matrix

# In[7]:


df.corr()


# ## Part 5: Display the Correlation Matrix as Heat Map

# In[8]:


fig, ax = plt.subplots(figsize=(15,15))
hm = sns.heatmap(df.corr(), fmt='.3f', cmap='bwr', annot=True, ax=ax, xticklabels='auto', yticklabels='auto')


# ## Part 5: Display the Correlation Matrix as Pair Plot

# In[9]:


selected_columns = df.columns[:10]
df_selected = df[selected_columns]
sns.pairplot(df_selected, diag_kind='kde')
plt.show()


# Based on the heat map and pair plot, the best guess predictor is `Ram`, followed by `SSD` and `Ppi`.

# ## Part 6: Simple linear regression

# Firstly let's split the data to X_train and X_test.

# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.2)
print("Length of X_train:", len(X_train))
print("Length of X_test:", len(X_test))


# In[11]:


model = smf.ols(formula='Price ~ Ram', data=X_train).fit()

print(model.summary())

adj_R2 = model.rsquared_adj
print("Adjusted R-squared value:", adj_R2)


# Then we create a Simple Linear Regression with formula `Price ~ SSD`.

# In[12]:


model = smf.ols(formula='Price ~ SSD', data=X_train).fit()

print(model.summary())

adj_R2 = model.rsquared_adj
print("Adjusted R-squared value:", adj_R2)


# Then we create a Simple Linear Regression with formula `Price ~ Ppi`.

# In[13]:


model = smf.ols(formula='Price ~ Ppi', data=X_train).fit()

print(model.summary())

adj_R2 = model.rsquared_adj
print("Adjusted R-squared value:", adj_R2)


# We can see that when we build the Simple Linear Regression Model with those 3 features that has highest corrlation with Price, the R-squared value was not large. 
# 
# `Price ~ Ram` has highest R-squared value of 0.483.
# 
# `Price ~ SSD` has highest R-squared value of 0.437.
# 
# `Price ~ Ppi` has highest R-squared value of 0.215.
# 
# Therefore we can build a Multi-Linear Regression Model to see if it can improve the accuracy.

# ## Part 7: Multi-Linear Regression Model

# In[14]:


# Initialize variables to store the best R² value and corresponding feature
best_r_squared = -1  # Start with the lowest possible value
best_predictor = ''

# Instantiate the regression model
model = LinearRegression()

# Loop through each column (except 'Price' as it's the target)
for column in df.columns:
    if column != 'Price' and df[column].dtype in ['float64', 'int64']:  # Ensure the column is numeric
        X = df[[column]]  # Feature matrix
        y = df['Price']  # Target variable

        # Fit the model
        model.fit(X, y)

        # Predict the mpg
        y_pred = model.predict(X)

        # Calculate R² value
        r_squared = r2_score(y, y_pred)

        # Check if this R² is the best we've seen so far
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_predictor = column

# Print the best predictor and its R² value
print(f'Best Predictor: {best_predictor}')
print(f'Best R² Value: {best_r_squared}')


# In[15]:


best_degree = 0
best_r_squared = 0

for degree in range(1, 21):  
    X = np.column_stack([np.power(df[best_predictor], i) for i in range(1, degree + 1)])
    y = df['Price']

    # Add a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Get the R² value
    r_squared = model.rsquared

    # Check if this R² is the best we've seen so far
    if r_squared > best_r_squared:
        best_r_squared = r_squared
        best_degree = degree

# Print the best degree and its R² value
print(f'Best Degree: {best_degree}')
print(f'Best R² Value: {best_r_squared}')
  


# In[16]:


model = smf.ols('Price ~ Ram + Weight + TouchScreen + Ips + Ppi + HDD + SSD', data=df).fit()

# Print the summary of the model
print(model.summary())


# From the summary we can see the some features have a p>0.05 so we can remove it from the formula.

# In[17]:


model = smf.ols('Price ~ Ram + Weight + Ips + Ppi + SSD', data=df).fit()

# Print the summary of the model
print(model.summary())


# Although the R-squared did not increase after applying the new formula, the less features included results a more simple model and avoid overfitting.

# #### Leverage vs. the Square of the Residual

# In[18]:


formula = 'Price ~ Ram + Weight + Ips + Ppi + SSD'
model = smf.ols(formula, data=df).fit()

# Calculate influences and residuals
influence = model.get_influence()
resid_squared = influence.resid_studentized_external ** 2
leverage = influence.hat_matrix_diag

# Plot leverage vs. squared residuals
plt.scatter(resid_squared, leverage)
plt.ylabel('Leverage')
plt.xlabel('Squared Residuals')
plt.title('Leverage vs. Squared Residuals')
plt.grid(True)
plt.show()


# # Conclusion
# 
# In this project, I aimed to identify the key factors influencing laptop prices using machine learning techniques. The process involved multiple steps, including data loading, cleaning, exploratory data analysis, and building linear regression model.
# 
# ## Key Findings
# 
# ### Data Preparation and Cleaning
# 
# We started by loading the dataset from Kaggle, ensuring data types were appropriate, and removing non-numeric and less relevant features. This step was crucial to prepare the data for analysis and modeling.
# 
# ### Correlation Analysis
# 
# Through the correlation matrix and visualizations like heat maps and pair plots, we can identified that RAM, SSD, and PPI had the highest correlations with the laptop prices. This guided us in selecting these features for further analysis.
# 
# ### Simple Linear Regression
# 
# We performed simple linear regression for each of the highly correlated features (RAM, SSD, and PPI). The RAM feature showed the highest adjusted R-squared value of 0.486, indicating a moderate level of predictive power.
# 
# ### Multi-Linear Regression Model
# 
# To improve the predictive accuracy, we built a multi-linear regression model using multiple features. The model initially included RAM, Weight, TouchScreen, Ips, Ppi, HDD, and SSD. We refined the model by removing features with high p-values (>0.05), leading to a final model with RAM, Weight, Ips, Ppi, and SSD. This simplified model reduced the risk of overfitting while maintaining a balance between complexity and predictive power.
# 
# ## Implications
# 
# The analysis highlighted that certain hardware specifications, particularly RAM and SSD, significantly impact laptop prices. These findings can assist manufacturers and consumers in understanding price determinants and making informed decisions. However, the predictive power of our models, while reasonable, suggests that other factors not included in the dataset might also play significant roles in determining laptop prices.
# 
# ### Future work could involve:
# 
# Expanding the dataset to include more features, such as brand reputation, build quality, and market trends.
# Utilizing advanced machine learning techniques like ensemble methods to enhance predictive accuracy.
# Conducting time-series analysis to understand how the impact of different features on price evolves over time.

# # GitHub Link
# 
# Here is the link of the GitHub Repository: https://github.com/chloefung/DTSA5509-Final-Project/

# In[ ]:




