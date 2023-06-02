# Industrail-Copper-Modelling

In this project we have used python with machine learning to predict the selling price in regression and Status in classification using Decision tree Regressor/Classifier,

we have created two files for this project one is the **main.py** and another one is the **streamit.py** 

Before that we need the below mentioned libarries to be imported
-------------------------------------------------------------------------**main.py**---------------------------------------------------------------------------------

**main.py**
```
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
```

**Regression - Predicting The Selling Price**

In this main.py we are doing outlier correction, handling the skewness, Encoding the categorical variable, Standardization now the data is ready to pass in the ML model but before that we need to split the dataset into train and test.

After splititng the data we are going to use **DecisionTreeRegressor** with **GridSearchCV** to find the **best _params_** then using mean_squared_error to find the
accuracy of our prediction.

once we find the best model we will pickle it in a file

Then we carete a UI Dashboard in streamlit to get the input from the users to predict the selling price 

Below is the Streamlit dashboard UI Screenshot.

![Screenshot (53)](https://github.com/SrinivasanGajendran/Industrail-Copper-Modelling/assets/46883734/c521e55d-4cd6-4928-804a-760c1cae684c)


**Classification - Predicting The Status (Won/Loss)**

Here, We are ecnoding the categorical varaibles and Applying **DecisionTreeClassifier** to find the status whether it is won/loss.
For this we will be using **Accuracy** for Evaluation.
Once we get the needed results we wil pickle it.

Below is the dashboard of it 

![Screenshot (54)](https://github.com/SrinivasanGajendran/Industrail-Copper-Modelling/assets/46883734/c3c64698-e85a-442b-8e06-88e224c16048)


**---------------------------------------------------------------Streamit.py-----------------------------------------------------------------------------**


For Regression, After getting the input we will unpickle it and pass the user entered values in it

```
    if submit_button:

        with open(r"D:\Assignments\Final_project/model.pkl", 'rb') as file:
            model = pickle.load(file)
        with open(r'D:\Assignments\Final_project/scaler.pkl', 'rb') as f:
            scale = pickle.load(f)

        with open(r"D:\Assignments\Final_project/ohe.pkl", 'rb') as f:
            one_hot = pickle.load(f)

        with open(r"D:\Assignments\Final_project/ohe1.pkl", 'rb') as f:
            one_hot_1 = pickle.load(f)

        new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
        new_sample_ohe = one_hot.transform(new_sample[:, [7]]).toarray()
        new_sample_be = one_hot_1.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scale.transform(new_sample)
        new_pred = model.predict(new_sample1)[0]
        st.write('## :green[Predicted selling price:] ', np.exp(new_pred))
```


For Classification, After getting the input we will unpickle it and pass the user entered values in it

```
        if submit_button:
            with open(r"D:\Assignments\Final_project/classifiermodel.pkl", 'rb') as file:
                clas_model = pickle.load(file)
            with open(r'D:\Assignments\Final_project/classifierscaler.pkl', 'rb') as f:
                clas_scale = pickle.load(f)

            with open(r"D:\Assignments\Final_project/classifieronehot.pkl", 'rb') as f:
                class_one_hot = pickle.load(f)

            new_sample = np.array([[np.log(float(quantity_tons)), application,np.log(float(selling)), np.log(float(thickness)), float(width),
                                    country, float(customer), int(product_ref), item_type]])
            new_sample_ohe = class_one_hot.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6,7 ]], new_sample_ohe), axis=1)
            new_sample1 = clas_scale.transform(new_sample)
            new_pred = clas_model.predict(new_sample1)
            if new_pred == 1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')
```

