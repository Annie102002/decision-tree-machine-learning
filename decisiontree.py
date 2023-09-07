#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[97]:


from sklearn.datasets import load_iris


# In[98]:


iris = load_iris()
iris


# In[99]:


iris.data


# In[100]:


iris.target


# In[1]:


dataset=pd.DataFrame(iris.data)
dataset


# In[102]:


dataset.columns=iris.feature_names
dataset


# In[103]:


x=dataset
y=iris.target


# In[104]:


x


# In[105]:


y


# In[106]:


from sklearn.model_selection import train_test_split


# In[107]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


# In[108]:


x_train


# In[109]:


y_train


# In[110]:


from sklearn.tree import DecisionTreeClassifier


# In[111]:


#postpruning technique
treemodel=DecisionTreeClassifier()


# In[112]:


treemodel 


# In[113]:


treemodel.fit(x_train,y_train)


# In[114]:


#constructing  tree
from sklearn import tree
plt.figure(figsize =(15,10))
tree.plot_tree(treemodel,filled=True)


# In[115]:


#postpruning technique
treemodel =DecisionTreeClassifier(max_depth=2)
treemodel


# In[116]:


treemodel.fit(x_train,y_train)


# In[117]:


#constructing tree
from sklearn import  tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)


# In[118]:


#prediction
y_pred=treemodel.predict(x_test)


# In[119]:


y_pred


# In[120]:


from sklearn.metrics import accuracy_score,classification_report


# In[121]:


score =accuracy_score(y_pred,y_test)
print(score)


# In[122]:


#prepruning
parameter ={'criterion':['gini','entropy','log_loss'],
             'splitter':['best','random'],
             'max_depth':[1,2,3,4,5],
             'max_features':['auto','sqrt','log2'],
             'ccp_alpha':[1,2,3,4,5,6,7]
            }


# In[123]:


from sklearn.model_selection import GridSearchCV


# In[124]:


treemodel=DecisionTreeClassifier()
cv=GridSearchCV(treemodel,param_grid=parameter,cv=5,scoring='accuracy')


# In[125]:


cv.fit(x_train,y_train)


# In[126]:


cv.best_params_


# In[127]:


cv.predict(x_test)


# In[128]:


y_pred=cv.predict(x_test)


# In[129]:


from sklearn.metrics import accuracy_score,classification_report


# In[130]:


score=accuracy_score(y_pred,y_test)


# In[131]:


score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




