import streamlit as st
import pandas as pd
from features.machine_learning import MinMaxScaler, ManhattanKNN
from io import BytesIO

def main():
    
    st.set_page_config("Prediksi Kualitas Udara Via File CSV atau Excel")
    
    st.title("Upload File CSV atau Excel ke Streamlit")

    st.subheader("Sebelum upload, pastikan terdapat kolom-kolom pada tabel seperti:")

    st.subheader("1. nama_kota")
    st.subheader("2. tanggal")
    st.subheader("3. pm10")
    st.subheader("4. pm2.5")
    st.subheader("5. so2")
    st.subheader("6. co")
    st.subheader("7. o3")
    st.subheader("8. no2")

    uploaded_file = st.file_uploader("Upload file CSV atau Excel file", type=["csv", "xlsx"])

    st.page_link("app.py", 
                 label="Kembali ke awal")

    if uploaded_file is not None:
        
        try:

            try:
                df = pd.read_csv(uploaded_file)

            except:
                df = pd.read_excel(uploaded_file)
            
            columns = {'nama_kota', 'tanggal', 'pm10', 'pm2.5',
                       'so2', 'co', 'o3', 'no2'}
            
            if columns & set(df.columns) != columns:
                st.error("Kolom masih belum sesuai! ")
                return 
            
            columns2 = ['pm10', 'pm2.5',
                        'so2', 'co', 'o3', 'no2']
            
            data = df[columns2].to_numpy()
            scaler = MinMaxScaler()
            model = ManhattanKNN()

            data = scaler.transform(data)
            df['label'] = model.predict(data)
            df['label'] = df['label'].replace({0:'BAIK',
                                               1:'SEDANG',
                                               2:'TIDAK SEHAT',
                                               3:'SANGAT TIDAK SEHAT'})
            
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, 
                      index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="Download sebagai CSV",
                data=csv_buffer,
                file_name=uploaded_file.name,
                mime="text/csv",
            )

            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, 
                        index=False, 
                        sheet_name="Sheet1")
            excel_buffer.seek(0)

            excel_file_name = f"{uploaded_file.name.rsplit('.', 1)[0]}.xlsx" \
                                if '.csv' in uploaded_file.name \
                                else uploaded_file.name

            st.download_button(
                label="Download sebagai Excel",
                data=excel_buffer,
                file_name=excel_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if st.button("Tampilkan Dashboard"):

                city_name = st.selectbox(
                            "Pilih kota berdasarkan data yang diinput.",
                            tuple(df['nama_kota'].unique()\
                                                .tolist()),
                        )
                
                eda_data = df[df['nama_kota'] == city_name]

                st.subheader(f"Grafik Sebaran Kategori Kualitas Udara Di {city_name}:")
                st.bar_chart(
                    data=eda_data.groupby('label')\
                        .agg({'so2':'count'})
                        .reset_index()\
                        .rename(columns={'so2':'count'}),
                    x='label',
                    y='count'
                )
                    
                st.subheader(f"Grafik Sebaran Rata-rata Data Kandungan Udara Dari Data Berdasarkan Kategori Di {city_name}:")
                grouped_data = eda_data.groupby(by=['label'])\
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

                st.subheader(f"Grafik Korelasi Keenam Polutan Udara Terhadap Kualitas Udara Di {city_name}")
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
            
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    main()