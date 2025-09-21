import streamlit as st
import pickle
import numpy as np

# Load model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Laptop Price Predictor 💻", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #4B0082;'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the price of your dream laptop instantly!</p>", unsafe_allow_html=True)
st.write("---")

# Input columns
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
    weight = st.number_input('Weight of Laptop (Kg)', 0.5, 5.0, 1.5, 0.1)
    touchscreen = st.radio('Touchscreen', ['No', 'Yes'])
    ips = st.radio('IPS Display', ['No', 'Yes'])

with col2:
    screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.3)
    resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
    hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
    ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
    os = st.selectbox('Operating System', df['os'].unique())

st.write("---")

# Prediction button
if st.button('Predict Price 💰'):
    # Convert inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # Query array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)

    # Prediction
    price = int(np.exp(pipe.predict(query)[0]))

    # Display result in a fancy box
    st.success(f"💵 Estimated Laptop Price: **₹ {price:,}**")
    st.balloons()
