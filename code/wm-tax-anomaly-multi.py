# With pyCaret
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from pycaret.anomaly import *

import plotly.graph_objects as go
import plotly.express as px

df = pd.read_csv('***')

#Since algorithms cannot directly consume date or timestamp data, 
# we will extract the features from the timestamp 
# and will drop the actual timestamp column before training models.

# set timestamp to index
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df.set_index('timestamp', drop=True, inplace=True)
# resample timeseries to hourly 
df.resample('H').sum()
print(df)

df.replace('n',0,inplace=True)
df.replace('y',1,inplace=True)
df.fillna(0,inplace=True)

#print(df.describe)
print(df.info)


# creature features from date
df['day'] = [i.day for i in df.index]
df['day_of_year'] = [i.dayofyear for i in df.index]
df['week_of_year'] = [i.weekofyear for i in df.index]
df['hour'] = [i.hour for i in df.index]
df.head()


#sns.boxplot(x='variable', y='value', data=pd.melt(df))
#sns.pairplot(df)

df.drop_duplicates(inplace=True)

plt.rcParams['figure.figsize']= (20,8)
sns.swarmplot(x='variable', y='value', data=pd.melt(df))
plt.show()

exp_1= setup(df,session_id=121, experiment_name='Anomaly1')

exp_2= setup(df,session_id=122, experiment_name='Anomaly2')
print(exp_2)
print(models())


# create multiple models

iforest=create_model('iforest')
knn=create_model('knn')
svm=create_model('svm')

# Detect Anomaly

iforest_result=assign_model(iforest)
knn_result=assign_model(knn)
svm_result=assign_model(svm)

print(iforest_result.head())



knn_anomaly=knn_result[knn_result['Anomaly']==1]
svm_anomaly=svm_result[svm_result['Anomaly']==1]
iforest_anomaly=iforest_result[iforest_result['Anomaly']==1]

print(svm_anomaly.head)
print("iforest_anomaly.shape")
print(iforest_anomaly.shape) 

print("knn_anomaly.shape")
print(knn_anomaly.shape)

print(knn_anomaly.head)

# let draw some digram

# plot value on y-axis and date on x-axis
fig = px.line(svm_result, x=svm_result.index, y="TAX_PCT", title='Walmart_Tax', template = 'plotly_dark')

# create list of outlier_dates
outlier_dates = svm_result[svm_result['Anomaly'] == 1].index
# obtain y value of anomalies to plot
y_values = [svm_result.loc[i]['TXN_TAXCODE'] for i in outlier_dates]

fig.add_trace(go.Scatter(x=outlier_dates, y=y_values, mode = 'markers', 
                name = 'Anomaly', 
                marker=dict(color='red',size=10)))

fig.show()

print("====Knn anomaly ====")
print(knn_anomaly)


print("====SVM anomaly ====")
print(svm_anomaly)









