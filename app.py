import streamlit as st
import os

st.page_link("app.py", 
             label="Home", 
             icon="🏠")

st.page_link("pages/page_1.py", 
             label="Prediksi Kualitas Udara Via Form", 
             icon="1️⃣")

st.page_link("pages/page_2.py", 
             label="Prediksi Kualitas Udara Via File CSV atau Excel", 
             icon="2️⃣")

st.page_link("pages/page_3.py", 
             label="Prediksi Kualitas Udara Dalam Periode Tahunan Pada Suatu Kota", 
             icon="3️⃣")