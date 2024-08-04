import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))

st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main"><h1>💻 Laptop Price Predictor</h1>', unsafe_allow_html=True)

st.markdown("### Enter the details of the laptop:")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique(), help="Select the brand of the laptop")
    laptop_type = st.selectbox('Type', df['TypeName'].unique(), help="Select the type of laptop")
    ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], help="Select the RAM size")
    weight = st.number_input("Weight (in kg)", help="Enter the weight of the laptop")

with col2:
    touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'], help="Does the laptop have a touchscreen?")
    ips = st.selectbox('IPS', ['No', 'Yes'], help="Does the laptop have an IPS display?")
    screen_size = st.number_input('Screen Size (in inches)', help="Enter the screen size")
    resolution = st.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
        '2560x1600', '2560x1440', '2304x1440'], help="Select the screen resolution")

cpu = st.selectbox('CPU', df['Cpu brand'].unique(), help="Select the CPU brand")
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048], help="Select the HDD size")
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024], help="Select the SSD size")
gpu = st.selectbox('GPU', df['Gpu brand'].unique(), help="Select the GPU brand")
os = st.selectbox('Operating System', df['os'].unique(), help="Select the operating system")

if st.button('Predict Price'):
    if screen_size == 0:
        st.error("Screen Size cannot be zero. Please enter a valid screen size.")
    else:
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        x_res, y_res = map(int, resolution.split('x'))
        ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size

        query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object).reshape(1, -1)

        predicted_price = np.exp(pipe.predict(query)[0])
        st.markdown(f"<div class='main'><h2>Predicted Price: ₹ {int(predicted_price)}</h2></div>", unsafe_allow_html=True)

