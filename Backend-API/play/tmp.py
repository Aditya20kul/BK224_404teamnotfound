import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle

df=pd.read_excel('data/HE_accidental.xls',index=False)
df.head()
rf=RandomForestRegressor()
X_N=df[['UA','LMTD','Efficiency_Accidental']]
y_N=df[['Tho','Tco']]
wrap = MultiOutputRegressor(rf)
wrap.fit(X_N,y_N)
wrap.score(X_N,y_N)
#print(wrap.predict([[0.19489, 43.13655, 90.83717]]))
filename = 'accidental_temp.sav'
pickle.dump(wrap, open(filename, 'wb'))