# Import libraries yang dibutuhkan
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Title aplikasi Streamlit
st.title("Prediksi Tingkat Obesitas")

# Deskripsi
st.write("Silakan masukkan data pasien untuk prediksi tingkat obesitas")

# Muat Model
with open("model_tuned.pkl", "rb") as f:
    model = pickle.load(f)


# Input data
age = st.number_input("Usia", min_value=0, max_value=100, value=25)
height = st.number_input("Tinggi Badan (m)", min_value=0.0, max_value=2.5, value=1.7)
weight = st.number_input("Berat Badan (kg)", min_value=0.0, max_value=500.0, value=70.0)
family_history = st.selectbox("Riwayat Obesitas Keluarga", ["yes", "no"])
favc = st.selectbox("Sering Mengonsumsi Makanan Tinggi Kalori?", ["yes", "no"])
fcvc = st.number_input("Jumlah Sayur per Hari", min_value=0, max_value=10, value=2)
ncp = st.number_input("Jumlah Makan per Hari", min_value=0, max_value=10, value=3)
caec = st.selectbox("Sering Mengemil?", ["Sometimes", "Frequently", "Always", "No"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
ch2o = st.number_input("Jumlah Air per Hari (L)", min_value=0.0, max_value=10.0, value=2.0)
scc = st.selectbox("Memantau Asupan Kalori?", ["yes", "no"])
faf = st.number_input("Frekuensi Aktivitas Fisik per Minggu", min_value=0.0, max_value=10.0, value=2.0)
tue = st.number_input("Jumlah Jam Menggunakan Perangkat Elektronik per Hari", min_value=0, max_value=20, value=4)
calc = st.selectbox("Sering Mengonsumsi Alkohol?", ["Sometimes", "Frequently", "Always", "No"])
mtrans = st.selectbox("Transportasi yang Digunakan", ["Automobile", "Motorbike", "Public Transportation", "Walking"])

# Tombol prediksi
if st.button("Prediksi"):

    # Mengumpulkan data yang diberika­n
    input_data = pd.DataFrame([[
        age, height, weight, family_history, favc, fcvc, ncp, 
        caec, smoke, ch2o, scc, faf, tue, calc, mtrans
    ]], columns=[
        'Age', 'Height', 'Weight', 'family_history_with_overweight',
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O',
        'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'
    ]) 


    # Pastikan urutan dan nama kolom sesuai training
    # Saya menyarankan untuk menyimpannya di pickle juga:
    with open("label_caec.pkl", "rb") as f:
        model_columns = pickle.load(f)

    # Reorder sesuai training
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Prediksi
    prediction = model.predict(input_data)[0]

    st.success(f"Prediksi Tingkat Obesitas : {prediction}")

