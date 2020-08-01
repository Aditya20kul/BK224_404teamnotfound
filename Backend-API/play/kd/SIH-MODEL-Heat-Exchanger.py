
import pandas as pd
import numpy as np
import random
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

df=pd.read_excel('HE_normal.xls')
A=44.6
# LMTD for counter-current flow
df['LMTD']= ((df['Thi'] - df['Tco']) - (df['Tho'] - df['Tci'])) / np.log((df['Thi'] - df['Tco']) / (df['Tho'] - df['Tci']) )

# to handle the anomaly of 0 denominator change Nan with approximate mean values

df.LMTD=df['LMTD'].fillna(35.2)
df['Effectiveness'] = abs((df['Tco'] - df['Tci']) / (df['Thi'] - df['Tci']))
df['Q_Ideal'] = df['UA']*A* df['LMTD']

l=[]
for i in range(len(df)):
    l.append(random.uniform(0.55,0.92)* df.Q_Ideal[i])
df['Q_Actual'] =l

df['Efficiency'] = df['Q_Actual'] / df['Q_Ideal']*100

X=df[['Q_Actual','Q_Ideal']]
y=df.Efficiency.values.reshape(-1,1)


from sklearn.svm import SVR
normal_svr=SVR(kernel='rbf')
normal_svr.fit(X,y)
normal_svr.score(X,y)*100
normal_svr.predict([[109.072635,159.875254]])

rf=RandomForestRegressor()
X_N=df[['UA','LMTD','Efficiency']]
y_N=df[['Tho','Tco']]
wrap = MultiOutputRegressor(rf)
wrap.fit(X_N,y_N)
wrap.score(X_N,y_N)
wrap.predict([[0.185,47.19,87.4]])


import random
df_foul = pd.read_excel('HE_normal.xls')
df_foul=df_foul.drop(columns=['Q_Actual','Efficiency'],axis=1)
l_foul=[]
for i in range(len(df_foul)):
    l_foul.append(random.uniform(0.30,0.53)* df_foul.Q_Ideal[i])
df_foul['Q_Maintenance'] =l_foul

df_foul['Efficiency_Maintenance'] = df_foul['Q_Maintenance'] / df_foul['Q_Ideal']*100
X=df_foul[['Q_Maintenance','Q_Ideal']]
y=df_foul.Efficiency_Maintenance.values.reshape(-1,1)

svr_foul=SVR(kernel='rbf')
svr_foul.fit(X,y)
svr_foul.score(X,y)*100


df_foul.to_excel('HE_Maintenance.xls')

svr_foul.predict([[172.533,374.116]])


rf=RandomForestRegressor()
X_M=df_foul[['UA','LMTD','Efficiency_Maintenance']]
y_M=df_foul[['Tho','Tco']]
wrap_M = MultiOutputRegressor(rf)
wrap_M.fit(X_M,y_M)
wrap_M.score(X_M,y_M)


wrap_M.predict([[0.185,47.19,36.797]])

# Accidental Situation data


df_accidental = pd.read_excel('HE_normal.xls')
df_accidental=df_accidental.drop(columns=['Q_Actual','Efficiency'],axis=1)

l_accidental=[]
for i in range(len(df_accidental)):
    l_accidental.append(random.uniform(0.1,0.30)* df_accidental.Q_Ideal[i])
df_accidental['Q_Accidental'] =l_accidental

df_accidental['Efficiency_Accidental'] = df_accidental['Q_Accidental'] / df_accidental['Q_Ideal']*100

X=df_accidental[['Q_Accidental','Q_Ideal']]
y=df_accidental.Efficiency_Accidental.values.reshape(-1,1)


svr_accidental=SVR(kernel='rbf')
svr_accidental.fit(X,y)
svr_accidental.score(X,y)*100

svr_accidental.predict([[52.85,327.7]])

rf=RandomForestRegressor()
X_A=df_accidental[['UA','LMTD','Efficiency_Accidental']]
y_A=df_accidental[['Tho','Tco']]
wrap_A = MultiOutputRegressor(rf)
wrap_A.fit(X_A,y_A)
wrap_A.score(X_A,y_A)

wrap_A.predict([[0.185,47.19,19.894]])


df_accidental.to_excel('HE_accidental.xls')


df_accidental.head()
#svr_accidental.predict([[3.411,17.758]])


