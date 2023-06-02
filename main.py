import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

def Industrail_Copper_Modelling():

    df = pd.read_excel('daily_offers.xlsx')
    df1 = df.copy()
#-----------------------Making sure the format is correct and changing null values to Nan--------------------------------------
    df1['item_date'] = pd.to_datetime(df1['item_date'], format='%Y%m%d', errors='coerce').dt.date
    df1['quantity tons'] = pd.to_numeric(df1['quantity tons'], errors='coerce')
    df1['customer'] = pd.to_numeric(df1['customer'], errors='coerce')
    df1['country'] = pd.to_numeric(df1['country'], errors='coerce')
    df1['application'] = pd.to_numeric(df1['application'], errors='coerce')
    df1['thickness'] = pd.to_numeric(df1['thickness'], errors='coerce')
    df1['width'] = pd.to_numeric(df1['width'], errors='coerce')
    df1.drop('material_ref', axis = 1, inplace=True)
    df1['product_ref'] = pd.to_numeric(df1['product_ref'], errors='coerce')
    df1['delivery date'] = pd.to_datetime(df1['delivery date'], format='%Y%m%d', errors='coerce').dt.date
    df1['selling_price'] = pd.to_numeric(df1['selling_price'], errors='coerce')

    df1 = df1.dropna()
    #-----------------------Creating a duplicate--------------------------
    df2 = df1.copy()
    #-----------------------Removing values less than or equal to Zero----

    lis = ['quantity tons','thickness','selling_price']
    for val in lis:
        dup = df2[val] <= 0
        #print(f'values less than zero in {val} :',dup.sum())
        df2.loc[dup, val] = np.nan
    df2.dropna(inplace=True)
    #------------------------Applying Log Transformation for the right skewed Data

    df2['selling_price'] = np.log(df2['selling_price'])
    df2['quantity tons'] = np.log(df2['quantity tons'])
    df2['thickness'] = np.log(df2['thickness'])

    #-------------------------Assigning X, Y-------------------------------------------

    X=df2[['quantity tons','status','item type','application','thickness','width','country','customer','product_ref']]
    y=df2['selling_price']

    #-------------------------Encoding Categorical Data-------------------------------------------

    ohe = OneHotEncoder()
    ohe.fit(X[['item type']])
    new_item_type = ohe.fit_transform(X[['item type']]).toarray()
    ohe1 = OneHotEncoder()
    ohe1.fit(X[['status']])
    new_status = ohe1.fit_transform(X[['status']]).toarray()

    #-------------------------Creatinga new X by using converted categorical and log transformed data-----------

    X = np.concatenate((X[['quantity tons', 'application', 'thickness', 'width','country','customer','product_ref']].values, new_item_type,new_status), axis=1)

    #--------------------------Applying Satndardization----------------------------------------

    scaler = StandardScaler()
    scaler.fit_transform(X)

    #---------------------------Splitting for train and test---------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    #----------------------------Using Decision Tree Regressor With GridsearchCV to find best params-------------------------------------

    param_grid = {
        'max_depth': [None, 5, 10],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }

    # Create a Decision Tree Regressor object
    dt_regressor = DecisionTreeRegressor(random_state=42)

    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=dt_regressor, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    best_regressor = DecisionTreeRegressor(random_state=42, **best_params)
    best_regressor.fit(X_train, y_train)
    y_pred = best_regressor.predict(X_test)

    #----------------------------Picklig the model----------------------
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_regressor, file)
    with open('scaler.pkl', 'wb') as file1:
        pickle.dump(scaler, file1)
    with open('ohe.pkl', 'wb') as file2:
        pickle.dump(ohe, file2)
    with open('ohe1.pkl', 'wb') as file3:
        pickle.dump(ohe1, file3)

    #-------------------Classification For predicting the status----------------------------------------------

    df_classifier = df2[df2['status'].isin(['Won', 'Lost'])]
    df_classifier['status'].unique()

    y_c = df_classifier['status']
    X_c = df_classifier[
        ['quantity tons', 'selling_price', 'item type', 'application', 'thickness', 'width', 'country', 'customer',
         'product_ref']]
    one = OneHotEncoder()
    X_one = one.fit_transform(X_c[['item type']]).toarray()
    one1 = LabelEncoder()
    one1.fit(y_c)
    y = one1.fit_transform(y_c)

    X = np.concatenate((X_c[['quantity tons', 'selling_price', 'application', 'thickness', 'width', 'country',
                             'customer', 'product_ref']].values, X_one), axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")
    new_sample = np.array([[np.log(800), np.log(856), 12, np.log(3), 1500, 28.0, 30113848, 1679998778, 'W']])
    new_sample_one = one.transform(new_sample[:, [8]]).toarray()
    new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_one), axis=1)
    new_sample = scaler.transform(new_sample)
    new_pred = dtc.predict(new_sample)
    #if new_pred == 1:
    #    print('The status is: Won')
    #else:
    #    print('The status is: Lost')

    with open('classifiermodel.pkl', 'wb') as file:
        pickle.dump(dtc, file)
    with open('classifierscaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('classifieronehot.pkl', 'wb') as f:
        pickle.dump(one, f)
