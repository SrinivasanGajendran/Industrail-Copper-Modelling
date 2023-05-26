import streamlit as st
import pandas as pd
import pickle
import numpy as np
st.set_page_config(layout="wide")

df = pd.read_excel('daily_offers.xlsx')
st.header('Industrial Copper Modelling Project')
tab1, tab2 = st.tabs(["REGRESSION", "CLASSIFICATION"])
with tab1:
    with st.form("my_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            status = st.selectbox("Status", df['status'].unique())
            item_type = st.selectbox("Item Type",df['item type'].unique())
            country = st.selectbox("Country", df['country'].unique())
            application = st.selectbox("Application", df['application'].unique())
            product_ref = st.selectbox("Product Reference",df['product_ref'].unique())
        with col3:
            quantity_tons = st.text_input("Enter Quantity Tons (Min:1 & Max:1000000000)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")


    if submit_button:

        with open(r"C:/Users/dines/Downloads/capstone_4/model.pkl", 'rb') as file:
            model = pickle.load(file)
        with open(r'C:/Users/dines/Downloads/capstone_4/scaler.pkl', 'rb') as f:
            scale = pickle.load(f)

        with open(r"C:/Users/dines/Downloads/capstone_4/ohe.pkl", 'rb') as f:
            one_hot = pickle.load(f)

        with open(r"C:/Users/dines/Downloads/capstone_4/ohe1.pkl", 'rb') as f:
            one_hot_1 = pickle.load(f)

        new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
        new_sample_ohe = one_hot.transform(new_sample[:, [7]]).toarray()
        new_sample_be = one_hot_1.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scale.transform(new_sample)
        new_pred = model.predict(new_sample1)[0]
        st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

with tab2:
    with st.form("my_form1"):
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            quantity_tons = st.text_input("Enter Quantity Tons (Min:1 & Max:1000000000)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            selling = st.text_input("Selling Price (Min:1, Max:100001015)")

        with col3:
            st.write(' ')
            item_type = st.selectbox("Item Type", df['item type'].unique(), key=21)
            country = st.selectbox("Country", df['country'].unique(), key=31)
            application = st.selectbox("Application", df['application'].unique(), key=41)
            product_ref = st.selectbox("Product Reference", df['product_ref'].unique(), key=51)
            submit_button = st.form_submit_button(label="PREDICT STATUS")

        if submit_button:
            with open(r"C:/Users/dines/Downloads/capstone_4/classifiermodel.pkl", 'rb') as file:
                clas_model = pickle.load(file)
            with open(r'C:/Users/dines/Downloads/capstone_4/classifierscaler.pkl', 'rb') as f:
                clas_scale = pickle.load(f)

            with open(r"C:/Users/dines/Downloads/capstone_4/classifieronehot.pkl", 'rb') as f:
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