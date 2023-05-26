import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


df = pd.read_excel('daily_offers.xlsx')
df1 = df.copy()

df1['item_date'] = pd.to_datetime(df1['item_date'], format='%Y%m%d', errors='coerce').dt.date
df1['quantity tons'] = pd.to_numeric(df1['quantity tons'], errors='coerce')
df1['customer'] = pd.to_numeric(df1['customer'], errors='coerce')
df1['country'] = pd.to_numeric(df1['country'], errors='coerce')
df1['application'] = pd.to_numeric(df1['application'], errors='coerce')
df1['thickness'] = pd.to_numeric(df1['thickness'], errors='coerce')
df1['width'] = pd.to_numeric(df1['width'], errors='coerce')
df1['material_ref'] = df1['material_ref'].str.lstrip('0')
df1['product_ref'] = pd.to_numeric(df1['product_ref'], errors='coerce')
df1['delivery date'] = pd.to_datetime(df1['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df1['selling_price'] = pd.to_numeric(df1['selling_price'], errors='coerce')

df1['material_ref'].fillna('unknown', inplace=True)
df1 = df1.dropna()

df2 = df1.copy()

#---------------------------------------------------------------------

lis = ['quantity tons','thickness','selling_price']
for val in lis:
    dup = df2[val] <= 0
    print(f'values less than zero in {val} :',dup.sum())
    df2.loc[dup, val] = np.nan
df2.dropna(inplace=True)
#--------------------------------------------------------------------

df2['selling_price'] = np.log(df2['selling_price'])
df2['quantity tons'] = np.log(df2['quantity tons'])
df2['thickness'] = np.log(df2['thickness'])

#--------------------------------------------------------------------

df2.drop('material_ref', axis = 1, inplace=True)

#--------------------------------------------------------------------

X=df2[['quantity tons','status','item type','application','thickness','width','country','customer','product_ref']]
y=df2['selling_price']

#--------------------------------------------------------------------

ohe = OneHotEncoder()
ohe.fit(X[['item type']])
new_item_type = ohe.fit_transform(X[['item type']]).toarray()
ohe1 = OneHotEncoder()
ohe1.fit(X[['status']])
new_status = ohe1.fit_transform(X[['status']]).toarray()

#-------------------------------------------------------------------

X = np.concatenate((X[['quantity tons', 'application', 'thickness', 'width','country','customer','product_ref']].values, new_item_type,new_status), axis=1)

#------------------------------------------------------------------

scaler = StandardScaler()
scaler.fit_transform(X)

#------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

#-----------------------------------------------------------------

Rfr = RandomForestRegressor(n_estimators=100, random_state=42)
Rfr.fit(X_train, y_train)
y_rf = Rfr.predict(X_test)
mse = mean_squared_error(y_test, y_rf)
print('mse',mse)
r2 = r2_score(y_test, y_rf)
print('r2',r2)

#-----------------------------------------------------------------
