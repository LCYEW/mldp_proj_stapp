import streamlit as st
import pandas as pd
from sklearn.linear_model import Ridge
import joblib




def user_input_features():
    open_price = st.sidebar.number_input('Open Price', 0.0, format = "%lf")
    high_price = st.sidebar.number_input('High Price', 0.0, format = "%lf")
    low_price = st.sidebar.number_input('Low Price', 0.0, format = "%lf")
    close_price = st.sidebar.number_input('Close Price', 0.0, format = "%lf")
    volume_btc = st.sidebar.number_input('Volume of BTC traded', 0.0, format = "%lf")
    date = st.sidebar.date_input("Date")
    data = {'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume_(BTC)': volume_btc,
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
           }
    features = pd.DataFrame(data, index=[0])
    return features
st.sidebar.header('User Input Parameters')

df = user_input_features()



model = joblib.load("trained_bitcoin_regressor_model.pkl")
price = model.predict(df)




st.write("""
# Bitcoin Price Estimator App
This app will take Bitcoin trading data and try to estimate the next price
""")

st.header(f"Predicted Weighted Price on {df.day}")
st.subheader(f":green[${round(price[0],2)}USD]")



st.subheader('User Input parameters')
st.write(df)


