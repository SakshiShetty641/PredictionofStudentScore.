#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - Data Science & Business Analytics Internship
# 
# # Task 1:- Prediction Using Supervised Machine Learning

# ### Name - Sakshi R Shetty

# In[ ]:


#Step1 - Importing all the Libraries required in this notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


#Step2 - Importing the data
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)
data.head(10)


# In[49]:


#Step3 - Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style = '+')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show


# In[47]:


#Step4- Dividing the data into Input & Output
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[36]:


#Step5- The Training Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[37]:


#Step6 - Training model using Linear Regression
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training Complete")


# In[38]:


#Step7- Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X, line);
plt.show()


# In[40]:


#Step9- Testing the data
print(X_test)
y_pred = regressor.predict(X_test)
                        


# In[42]:


#Step10 - Comparison
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[50]:


#Step11 Final Result - The predicted score if the student studies for 9.25hrs/day.
pred=regressor.predict(np.array([9.25]).reshape(1,1))
print("The predicted score is",pred)

