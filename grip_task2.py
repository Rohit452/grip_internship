#!/usr/bin/env python
# coding: utf-8

# ## Simple linear regression :
# In this regression task ,i need to predict the percentage of marks that student is expected to score if a student studies for 9.25 hrs a day.It's a simple regression task as it involves just two variables ie. no. of hrs and corresponding score.
# 
# INDEX
# 1. Importl libraries and dataset
# 2. Exploring the Data
# 3. Preparing and visualising the dataset
# 4. Training the Algorithm
# 5. Visualize comparison of result
# 6. Predicting the value on Test set
# 7. Making Predictions For Given Question
# 8. Evaluating result

# In[1]:


#import libraries for data wrangling ,scientinfic computing and plotting
import numpy as np                 
import pandas as pd                
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib.inline', '')


# In[2]:


# import data
url="task2_data.txt"
data = pd.read_csv(url)


# In[3]:


# exploring dataset
data.head()


# In[4]:


data.corr()


# In[5]:


data.describe()


# In[6]:


import seaborn as sns
import seaborn as seabornInstance


# In[7]:



seabornInstance.set(rc={"figure.figsize": (10, 4)})
figure,axes=plt.subplots(1,2)
seabornInstance.distplot(data['Scores'], bins=5,ax=axes[0])

seabornInstance.distplot(data['Hours'], bins=5,ax=axes[1])


# In[8]:


#visualizing data
data.plot(x="Hours",y="Scores",style="o")
plt.title("scatter plot between hours and score")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.show()


# In[9]:


#Preparing dataset
x=data.iloc[:,0:1].values
y=data.iloc[:,1].values


# In[10]:


# training the algorithm
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[11]:



from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x,y)
r_sq=model.score(x,y)
print("coefficient of determination:" ,r_sq)


# In[12]:


#visualising result
regression=model.coef_*x+model.intercept_
plt.scatter(x,y)
plt.plot(x,regression)
plt.show()


# In[13]:


#predicting value of test set
y_pred=model.predict(x_test)
new_data=pd.DataFrame({"Actual": y_test,"predicted": y_pred})
print(new_data)


# In[14]:


#making prediction for given question
hr=np.array([9.25])
prediction=model.predict(hr.reshape(-1,1))
print("no. of hours: ",hr[0])
print("percentage score: ",prediction[0])


# In[15]:


#performance evaluation
from sklearn import metrics
#mean absolute error
print("MAE: ", metrics.mean_absolute_error(y_test,y_pred))
#mean squared error
print("MSE: ",metrics.mean_squared_error(y_test,y_pred) )
#root mean square error
print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

