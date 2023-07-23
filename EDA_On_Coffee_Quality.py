#!/usr/bin/env python
# coding: utf-8

# # <center> EDA on Coffee Quality Dataset </center>

# ![image.png](attachment:image.png)

# ###  Identification of attributes and implementing the DataSet

# In[6]:


#Importing necessary libraries to perform EDA
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


#To read/open csv file
df=pd.read_csv("df_Coffee.csv")
print("First/Top 2 records>>>")
#Top 2 records serially
df.head(2)


# In[9]:


#Breif information of all attributes
df.info()


# In[10]:


#Statistical infromation of the given coffee data set attribute
df.describe()


# In[11]:


# To count the total rows and columns present in the data set
df.shape


# In[12]:


# Column names 
df.columns


# In[13]:


# Determining the start and the stop index
df.index


# ### Observation
# 
#   * In this block,there were basic syntax to get a brief overview of the dataset.Here are few conclusions drawn:
#   -There are 207 rows and 40 columns.
#   -Also drawn the statistical inference.
#   -The index starts from 0 and goes till 207.

# ### DataCleaning

# ![image.png](attachment:image.png)

# In[42]:


#Removing Unwanted column
df.drop('Unnamed: 0',axis=1)


# In[46]:


#To rename a column from bag_weight to bag_weight\kg
df1=df.rename(
    columns={
        'bag_weight': 'bag_weight\kg'
    })


# In[46]:


# Top 5 rows from df1 dataSet
df1.head()


# In[15]:


dfn=df1.select_dtypes(exclude='object')


# In[16]:


# To check whether the given rows have duplicate values
dfn.duplicated()


# In[17]:


dfn.duplicated().sum()


# In[18]:


# To check whether the given rows have null values
dfn.isnull()


# In[19]:


dfn.isnull().sum()


# ### Observations
#     -There existed a column with no significance 'Unnamed: 0',that has been dropped.
#     -All the values are checked whether there are null present in the dataset,but none of them has null values.
#     -All the values are checked whether there are duplicated present in the dataset,but none of them has duplicated values.

# ### EDA on coffee data set

# In[20]:


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()


# In[48]:


cat_list = df1.select_dtypes(include=["object"]).columns.tolist()
for col in cat_list: 
    plt.figure(figsize=(8,5))
    top10 = df1[col].value_counts()[:10]
    top10.plot(kind='bar')
    plt.title("Top 10 " + col)
    plt.grid(visible=False)
    plt.show()


# In[21]:


# company wise count
df1['Company'].value_counts()#Return a Series containing counts of unique rows in the DataFrame.


# In[65]:


# Variety wise count
df1['Variety'].value_counts()#Return a Series containing counts of unique rows in the DataFrame.


# In[22]:


# color wise count
df1['Color'].value_counts()#Return a Series containing counts of unique rows in the DataFrame.


# In[56]:


df1.groupby('Country of Origin')['Country of Origin'].agg('count')

#A groupby operation involves some combination of splitting the
# object, applying a function, and combining the results.


# In[35]:


#Creating a GRAPH to analyze AROMA distribution
ax = df1['Aroma'].plot(kind='hist',bins=20,title= 'Aroma Score')
ax.set_xlabel("Aroma Score")
plt.show()


# ### observation
#     - Aroma score,i.e,between 7.5 to 8.0, has the highest frequency.

# In[33]:


#Creating a GRAPH to analyze Flavour distribution
ax = df1['Flavor'].plot(kind='hist', bins=20, title= 'Flavor Score')
ax.set_xlabel("Flavor Score")
plt.show()


# ### Observation
#     -  Flavor score,i.e,between 7.75 to 8.0, has the highest frequency.

# In[34]:


#Creating a GRAPH to analyze After Taste distribution
ax = df1['Aftertaste'].plot(kind='hist', bins=20, title= 'Aftertaste Score')
ax.set_xlabel("Aftertaste Score")
plt.show()


# ### Observation
#     - Aftertaste score,i.e,between 7.50 to 7.75, has the highest frequency.

# In[36]:


#Creating a GRAPH to analyze Acidity Analysis distribution
ax = df1['Acidity'].plot(kind='hist', bins=20, title= 'Acidity Score')
ax.set_xlabel("Acidity Score")
plt.show()


# ### Observation
#     - Acidity score,i.e,between 7.50 to 8.0, has the highest frequency.

# In[37]:


#Creating a GRAPH to analyze Sweetness Analysis distribution
ax = df1['Sweetness'].plot(kind='hist', bins=20, title= 'Sweetness Score')
ax.set_xlabel("Sweetness Score")
plt.show()


# ### Observation
#     - Sweetness score is 10 for frequency 200+,this signifies all the flavours/coffee has sweetness score 10.

# In[38]:


#Creating a GRAPH to analyze Overall Analysis distribution
ax = df1['Overall'].plot(kind='hist', bins=20, title= 'Overall Score')
ax.set_xlabel("Overall Score" )
plt.show()


# ### Observation
#     - Overall score,i.e,between 7.50 to 8.0, has the highest frequency.

# In[39]:


#Creating a GRAPH to analyze Defects Analysis distribution
ax = df1['Defects'].plot(kind='hist', bins=20, title= 'Defects Score')
ax.set_xlabel("Defects Score")
plt.show()


# ### Observation
#     - Defects score is 0.0 for 200+ frequencies, this  signifies all the rows have no defects.

# In[40]:


#Creating a GRAPH to analyze Total Cup Points distribution
ax = df1['Total Cup Points'].plot(kind='hist', bins=20, title= 'Total Cup Points Score')
ax.set_xlabel("Total Cup Points Score")
plt.show()


# ### Observation
#     - Total cup points score,i.e,from 82 to 86, has the highest frequency.

# In[53]:


# Plotting a scatter plot of flavor and aftertaste scores
plt.figure(figsize=(8, 6))

# Plotting the scatter plot
plt.scatter(df1['Flavor'], df1['Aftertaste'])

# Adding labels and title
plt.xlabel('Flavor')
plt.ylabel('Aftertaste')
plt.title('Scatter Plot of Flavor and Aftertaste Scores')

# Displaying the plot
plt.show()


# In[47]:


# Determining the countries Maximum and the Minimum total cup score
mean_country_cupscore = df1.groupby('Country of Origin')['Total Cup Points'].mean().reset_index().head(10)
mean_country_cupscore.sort_values('Total Cup Points', ascending=False)


# ### Observations
#     -Ethiopia has the highest total cup  score with mean score 84.960909.
#     -El Salvador has the lowest total cup  score with mean score 81.532857.
# 
