#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
import seaborn as sns


# In[2]:



# Importing dataset and examining 
ESdata = pd.read_csv("EireStay.csv")
print(ESdata.head())
print(ESdata.shape)
print(ESdata.info())
print(ESdata.describe())


# In[3]:


# Converting Categorical features into Numerical 

ESdata['meal'] = ESdata['meal'].map({'BB':0, 'FB':1, 'HB':2,'SC':3, 'Undefined':4 })
ESdata['market_segment'] = ESdata['market_segment'].map({'Direct':0, 'Online TA':1})
ESdata['reserved_room_type'] = ESdata['reserved_room_type'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8})
ESdata['deposit_type'] = ESdata['deposit_type'].map({'No Deposit':0, 'Non Refund':1, 'Refundable': 2})

print(ESdata.info())


# In[4]:




# Drawing the Correlation Heatmap to find out the correlation and infer causation
corrs = ESdata.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')


# In[15]:



#lead_time=lt
#stays_in_weekend_nights=stayweekend
#stays_in_week_nights=stayweeknight
#adults=adult
#children=cld
#babies=bb
#meal=ml
#market_segment=ms
#previous_stays=ps
#reserved_room_type=rrt
#booking_changes=bookingc
#deposit_type=dt
#days_in_waiting_list=diwl
#average_daily_rate=adr
#total_of_special_requests=tosr


# Dividing data into 4 subsets
#Subset 1
subset1 = ESdata[['adults','previous_stays','reserved_room_type','deposit_type']]

#Subset 2
subset2 = ESdata[['stays_in_weekend_nights','stays_in_week_nights','adults','average_daily_rate','children','babies']]

#Subset 3
subset3 = ESdata[['adults','total_of_special_requests','meal', 'days_in_waiting_list']]

#Subset 4
subset4 = ESdata[['market_segment', 'adults','lead_time','booking_changes']]








# In[16]:



# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3 = feature_scaler.fit_transform(subset3)
X4 = feature_scaler.fit_transform(subset4)


# In[17]:



# Analysis on subset1 - bookings done previously. 
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[18]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 6)
kmeans.fit(X1)


# In[19]:



#subset1 = ESdata[['adults','previous_stays','reserved_room_type','deposit_type']]

# Implementing t-SNE to visualize the subset 
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

adult = list(ESdata['adults'])
ps = list(ESdata['previous_stays'])
rrt = list(ESdata['reserved_room_type'])
dt = list(ESdata['deposit_type'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color= kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'adults: {a}; previous_stays: {b}; reserved_room_type:{c}, deposit_type:{d}; ' for a,b,c,d, in list(zip(adult,ps,rrt,dt))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# In[20]:


# Analysis on subset2 
# Finding the number of clusters (K) - Elbow Plot 
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[21]:



# Running KMeans to generate labels for the subset
kmeans = KMeans(n_clusters = 6)
kmeans.fit(X2)


# In[22]:



#subset2 = ESdata[['stays_in_weekend_nights','stays_in_week_nights','adults','average_daily_rate','children','babies']]

# Implementing t-SNE to visualize dataset 2
tsne = TSNE(n_components = 2, perplexity =30,n_iter=5000)
x_tsne = tsne.fit_transform(X2)

stayweekend = list(ESdata['stays_in_weekend_nights'])
stayweeknight = list(ESdata['stays_in_week_nights'])
adult = list(ESdata['adults'])
adr = list(ESdata['average_daily_rate'])
cld = list(ESdata['children'])
bb = list(ESdata['babies'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'stays_in_weekend_nights: {a}; stays_in_week_nights: {b}; adults:{c}, average_daily_rate:{d}, children:{e},babies:{f}' for a,b,c,d,e,f in list(zip(stayweekend,stayweeknight,adult,adr,cld,bb))],
                                hoverinfo='text')]



layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE2.html')


# In[23]:


# Analysis on subset3 
# Finding the number of clustersK- Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)
    
    
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[24]:



# Running KMeans to create labels
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X3)


# In[25]:



#Subset 3

#subset3 = ESdata[['adults','total_of_special_requests','meal', 'days_in_waiting_list']]


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X3)

adult = list(ESdata['adults'])
tosr = list(ESdata['total_of_special_requests'])
ml = list(ESdata['meal']) 
diwl = list(ESdata['days_in_waiting_list'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'adults: {a}; total_of_special_requests: {b}; meal:{c}, days_in_waiting_list:{d}, ' for a,b,c,d, in list(zip(adult,tosr,ml,diwl))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE3.html')


# In[26]:


# Analysis on subset4
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X4)
    inertia.append(kmeans.inertia_)
    
    
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[27]:



# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X4)


# In[28]:



#Subset 4
#subset4 = ESdata[['market_segment', 'adults','lead_time','booking_changes']]


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X4)

ms = list(ESdata['market_segment'])
adult = list(ESdata['adults'])
lt = list(ESdata['lead_time'])
bookingc = list(ESdata['booking_changes']) 


data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'market_segment: {a}; adults: {b}; lead_time:{c}, booking_changes:{d} ' for a,b,c,d in list(zip(ms,adult,lt,bookingc))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE4.html')


# In[ ]:





# In[ ]:




