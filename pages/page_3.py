import streamlit as st
import pandas as pd
import requests
import datetime
from features.machine_learning import ManhattanKNN, MinMaxScaler
from io import BytesIO

import os

def main():

    def get_hourly_air_pollution(latitude, 
                                 longitude, 
                                 start_ts, 
                                 end_ts):

        api_key = st.secrets.get("API_KEY") or os.getenv("API_KEY")
        url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={latitude}&lon={longitude}&start={start_ts}&end={end_ts}&appid={api_key}"
        response = requests.get(url)
        response.raise_for_status()

        data = response.json().get('list', [])
        return data

    def process_hourly_to_daily(hourly_data):

        if not hourly_data:
            return None
        
        data = [i['components'] for i in hourly_data]
        time_data = [i['dt'] for i in hourly_data]

        df = {'time':time_data,
            'pm10':[i['pm10'] for i in data],
            'pm2.5':[i['pm2_5'] for i in data],
            'so2':[i['so2'] for i in data],
            'co':[i['co'] for i in data],
            'o3':[i['o3'] for i in data],
            'no2':[i['no2'] for i in data]}
        
        df = pd.DataFrame(df)

        if not df.empty:

            df['time'] = pd.to_datetime(df['time'], 
                                        unit='s', 
                                        utc=True)\
                        .dt.date
            df = df.groupby('time')\
                .agg({'pm10':'mean',
                        'pm2.5':'mean',
                        'so2':'mean',
                        'co':'mean',
                        'o3':'mean',
                        'no2':'mean'})
            
            return df
        
        return None

    def datetime_to_unix(dt):
        return int(dt.timestamp())
    
    csv_path = 'lat_long.csv'
    provinces = pd.read_csv(csv_path)

    province = st.selectbox(
                        "Pilih berdasarkan nama provinsi.",
                        tuple(provinces['name'].unique()\
                                               .tolist()),
                    )
    
    year_time = st.selectbox(
                        "Pilih periode berdasarkan tahun:",
                        tuple(range(2021, 2025, 1)),
                    )
    
    start = datetime.datetime(year_time, 1, 1, 0, 0, 0, 
                              tzinfo=datetime.timezone.utc)
    end = datetime.datetime(year_time, 12, 31, 23, 59, 59, 
                            tzinfo=datetime.timezone.utc)

    start_ts = datetime_to_unix(start)
    end_ts = datetime_to_unix(end)

    province_row = provinces[provinces['name'] == province]
    lat, long = province_row['latitude'].to_list()[0], province_row['longitude'].to_list()[0]

    data = get_hourly_air_pollution(lat,
                                    long,
                                    start_ts,
                                    end_ts)
    
    data = process_hourly_to_daily(data)
    columns2 = ['pm10', 'pm2.5',
                'so2', 'co', 'o3', 'no2']
            
    data_to_predict = data[columns2].to_numpy()
    scaler = MinMaxScaler()
    model = ManhattanKNN()

    data_to_predict = scaler.transform(data_to_predict)
    data['label'] = model.predict(data_to_predict)
    data['label'] = data['label'].replace({0:'BAIK',
                                            1:'SEDANG',
                                            2:'TIDAK SEHAT',
                                            3:'SANGAT TIDAK SEHAT'})

    st.subheader(f"Grafik Sebaran Kualitas Udara Dari Tahun {year_time} Di {province}:")
    st.bar_chart(
    data=data.groupby('label')\
            .agg({'so2':'count'})
            .reset_index()\
            .rename(columns={'so2':'count'}),
        x='label',
        y='count'
    )
        
    st.subheader(f"Grafik Sebaran Rata-rata Data Kandungan Udara Dari Data Berdasarkan Kategori Pada Tahun {year_time} Di {province}:")
    grouped_data = data.groupby(by=['label'])\
                    .agg(dict(
                            zip(
                                ['pm10', 'pm2.5', 'so2', 
                                'co', 'o3', 'no2'],
                                ['mean'] * 6
                                )
                            )
                    )\
                    .reset_index()

    air_parameters = []
    values = []
    labels = []

    for i in ['pm10', 'pm2.5', 'so2', 
                'co', 'o3', 'no2']:
        
        temp_data = grouped_data[[i, 'label']]
        air_parameters += [i] * temp_data.shape[0]
        values += temp_data[i].tolist()
        labels += temp_data['label'].tolist()

    grouped_data = pd.DataFrame({'params':air_parameters,
                                    'values':values,
                                    'labels':labels})
    
    grouped_data['labels'] = grouped_data['labels'].replace({'BAIK':'0:BAIK',
                                                                'SEDANG':'1:SEDANG',
                                                                'TIDAK SEHAT':'2:TIDAK SEHAT',
                                                                'SANGAT TIDAK SEHAT':'3:SANGAT TIDAK SEHAT'})

    st.bar_chart(data=grouped_data, 
                    x="params", 
                    y="values", 
                    color="labels", 
                    stack=False,
                    horizontal=False)

    st.subheader(f"Grafik  Korelasi Keenam Polutan Udara Terhadap Kualitas Udara Di {province} Tahun {year_time}")
    corr_df = df[['pm10','pm2.5','so2','co',
                              'o3','no2','label']]\
                .replace({'BAIK':0,
                          'SEDANG':1,
                          'TIDAK SEHAT':2,
                          'SANGAT TIDAK SEHAT':3})\
                .corr(method='pearson')[['label']].head(-1)\
                .reset_index()
    
    st.bar_chart(
            corr_df,
            x='index',
            y='label'
        )
    
    df = data.copy()\
             .reset_index()\
             .rename(columns={'time':'tanggal'})
    df['provinsi'] = [province] * df.shape[0]
    df = df[['provinsi', 'tanggal', 'pm10', 'pm2.5',
             'so2', 'co', 'o3', 'no2', 'label']]

    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, 
                index=False)
    csv_buffer.seek(0)

    st.download_button(
        label="Download sebagai CSV",
        data=csv_buffer,
        file_name=f"{province}.csv",
        mime="text/csv",
    )

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, 
                index=False, 
                sheet_name="Sheet1")
    excel_buffer.seek(0)

    st.download_button(
        label="Download sebagai Excel",
        data=excel_buffer,
        file_name=f"{province}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if __name__ == "__main__":
    main()