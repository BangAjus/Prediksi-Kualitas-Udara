import streamlit as st
from features.machine_learning import MinMaxScaler, ManhattanKNN
import numpy as np
import requests
import pandas as pd

from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("API_KEY")

def main():

    st.title("Fitur Prediksi Data Kualitas Udara")
    
    subfeature = st.radio(
                        "Pilih Subfitur",
                        ["Kualitas Udara Sekarang di Provinsi Tertentu",
                         "Input Pribadi"],
                        captions=[
                            "Memprediksi kualitas udara berdasarkan nama provinsi",
                            "Input data yang diperlukan secara pribadi"
                        ],
                    )
    
    if subfeature == 'Kualitas Udara Sekarang di Provinsi Tertentu':

        csv_path = 'C:/Gabut/Flask/Finale/apps/lat_long.csv'
        data = pd.read_csv(csv_path)

        def get_air_pollution_data(latitude, 
                               longitude):
        
            base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': st.secrets.get("API_KEY") or os.getenv("API_KEY")
            }

            try:

                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                return data['list'][0]['components']
            
            except requests.exceptions.RequestException as e:

                print(f"Error fetching air pollution data: {e}")
                return None
        
        option = st.selectbox(
                        "Pilih berdasarkan nama provinsi.",
                        tuple(data['name'].tolist()),
                    )

        if st.button("Prediksi"):

            lat = data[data['name'] == option]['latitude']
            long = data[data['name'] == option]['longitude']

            data = get_air_pollution_data(lat, long)

            category = {0:'BAIK',
                        1:'SEDANG',
                        2:'TIDAK SEHAT',
                        3:'SANGAT TIDAK SEHAT'}
            
            model = ManhattanKNN()
            scaler = MinMaxScaler()

            data = scaler.transform(np.array([[data['pm10'],
                                            data['pm2_5'],
                                            data['so2'],
                                            data['co'],
                                            data['o3'],
                                            data['no2']]]))
            
            prediction = category[model.predict(data)[0]]

            st.write("---")

            st.write(f"**Nama Kota:** {option}")

            st.write(f"**Nilai PM10 (µg/m³):** {data[0][0]} ")
            st.write(f"**Nilai PM2.5 (µg/m³):** {data[0][1]}")
            st.write(f"**Nilai SO2 (µg/m³):** {data[0][2]}")

            st.write(f"**Nilai CO (µg/m³):** {data[0][3]}")
            st.write(f"**Nilai O3 (µg/m³):** {data[0][4]}")
            st.write(f"**Nilai NO2 (µg/m³):** {data[0][5]}")

            st.write(f"**Hasil Prediksi:** {prediction}")

            st.write("---")
            
            st.write("Berhasil melakukan prediksi!") 

    elif subfeature == 'Input Pribadi':

        city_name = st.text_input("Nama Kota:")

        pollutant1 = st.number_input("Masukkan nilai untuk PM10:", 
                                    step=0.1)
        pollutant2 = st.number_input("Masukkan nilai untuk PM2.5:", 
                                    step=0.1)
        pollutant3 = st.number_input("Masukkan nilai untuk SO2:", 
                                    step=0.1)
        
        pollutant4 = st.number_input("Masukkan nilai untuk CO:", 
                                    step=0.1)
        pollutant5 = st.number_input("Masukkan nilai untuk O3:", 
                                    step=0.1)
        pollutant6 = st.number_input("Masukkan nilai untuk NO2:", 
                                    step=0.1)

        if st.button("Prediksi"):
            
            category = {0:'BAIK',
                        1:'SEDANG',
                        2:'TIDAK SEHAT',
                        3:'SANGAT TIDAK SEHAT'}
            
            model = ManhattanKNN()
            scaler = MinMaxScaler()

            data = scaler.transform(np.array([[pollutant1,
                                            pollutant2,
                                            pollutant3,
                                            pollutant4,
                                            pollutant5,
                                            pollutant6]]))
            
            prediction = category[model.predict(data)[0]]

            st.write("---")

            st.write(f"**Nama Kota:** {city_name}")

            st.write(f"**Nilai PM10:** {pollutant1}")
            st.write(f"**Nilai PM2.5:** {pollutant2}")
            st.write(f"**Nilai SO2:** {pollutant3}")

            st.write(f"**Nilai CO:** {pollutant4}")
            st.write(f"**Nilai O3:** {pollutant5}")
            st.write(f"**Nilai NO2:** {pollutant6}")

            st.write(f"**Hasil Prediksi:** {prediction}")

            st.write("---")
            
            st.write("Data telah di-submit dan sukses diprediksi") 

if __name__ == "__main__":
    main()