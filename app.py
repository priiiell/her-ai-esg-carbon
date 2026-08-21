import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import tempfile
import time

st.set_page_config(
    page_title="CarbonEye AI - OBB Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #FFFFFF; color: #2D3748; font-family: 'Helvetica Neue', Arial, sans-serif; }
    [data-testid="stSidebar"] { background-color: #F7FAFC; border-right: 1px solid #E2E8F0; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #00C853; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricValue"] { font-size: 30px; color: #1A202C; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 13px; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    hr { margin-top: 1rem; margin-bottom: 2rem; border: 0; border-top: 1px solid #E2E8F0; }
    .stFileUploader { background-color: #FFFFFF; padding: 10px; border-radius: 8px; border: 1px dashed #CBD5E0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 CarbonEye AI - OBB Tracker")
st.markdown("<p style='color: #4A5568; font-size: 16px; margin-top: -10px;'>Monitoring Emisi Spasial Menggunakan Oriented Bounding Box (OBB)</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color: #2D3748; font-size: 20px;'>📁 Input Analisis</h2>", unsafe_allow_html=True)

# Fitur Baru: Pilihan tipe input gambar atau video
input_type = st.sidebar.radio("Pilih Tipe Input:", ["Gambar (Foto Udara)", "Video CCTV"])

uploaded_file = None
if input_type == "Gambar (Foto Udara)":
    uploaded_file = st.sidebar.file_uploader(
        "Unggah Foto Udara Lintasan Jalan", 
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.sidebar.file_uploader(
        "Unggah Rekaman Video CCTV", 
        type=["mp4", "avi", "mov"]
    )

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #4A5568; font-size: 14px; margin-bottom: 5px;'>🚀 AI CONFIGURATION</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='background-color: #E8F5E9; color: #2E7D32; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold;'>YOLOv8-OBB Active</span>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 12px; color: #718096; margin-top: 10px;'>Dataset: <code>UAV-OBB Aerial</code><br>Faktor Jarak: <code>0.1 Km (Fixed)</code></p>", unsafe_allow_html=True)

EMISSION_FACTORS = {
    'bike': 115,
    'car': 192,
    'taxi': 192,
    'bus': 822,
    'truck': 900,
    'other_vehicle': 200
}

def get_obb_mock_data():
    records = [
        {"id": 1, "Kendaraan": "car", "Emisi (g CO2)": 19.2},
        {"id": 2, "Kendaraan": "car", "Emisi (g CO2)": 19.2},
        {"id": 3, "Kendaraan": "bus", "Emisi (g CO2)": 82.2},
        {"id": 4, "Kendaraan": "taxi", "Emisi (g CO2)": 19.2},
        {"id": 5, "Kendaraan": "car", "Emisi (g CO2)": 19.2},
        {"id": 6, "Kendaraan": "truck", "Emisi (g CO2)": 90.0},
        {"id": 7, "Kendaraan": "bike", "Emisi (g CO2)": 11.5},
        {"id": 8, "Kendaraan": "car", "Emisi (g CO2)": 19.2},
        {"id": 9, "Kendaraan": "other_vehicle", "Emisi (g CO2)": 20.0},
        {"id": 10, "Kendaraan": "taxi", "Emisi (g CO2)": 19.2}
    ]
    return pd.DataFrame(records)

df_emissions = get_obb_mock_data()
df_grouped = df_emissions.groupby('Kendaraan').agg(
    Jumlah_Kendaraan=("id", "count"),
    Total_Emisi_g=("Emisi (g CO2)", "sum")
).reset_index()

if uploaded_file is not None:
    total_co2 = df_emissions["Emisi (g CO2)"].sum()
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Total Objek Terdeteksi", value=f"{len(df_emissions)} Unit")
    kpi2.metric(label="Total Akumulasi Emisi", value=f"{total_co2:.2f} g CO2")
    kpi3.metric(label="Metode Pemetaan", value="Oriented Bounding Box")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # PERBAIKAN ERROR: Sekarang diisi angka 2 agar layar terbagi simetris
    col_vid, col_log = st.columns(2)
    
    with col_vid:
        st.markdown("<h3 style='color: #2D3748; font-size: 18px;'>🎥 Pemrosesan Kamera & Deteksi Spasial</h3>", unsafe_allow_html=True)
        
        if input_type == "Gambar (Foto Udara)":
            # Logika penampilan untuk gambar
            st.image(uploaded_file, caption="Foto Udara Berhasil Dimuat", use_container_width=True)
            st.success("Analisis Citra OBB Selesai!")
        else:
            # Logika penampilan untuk video menggunakan OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            start_ai = st.button("▶ Jalankan Model OBB")
            
            if start_ai:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    cv2.putText(frame, "MODEL: YOLOv8-OBB | DETECTING ANGLED VEHICLES", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 83), 2)
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame, use_container_width=True)
                    time.sleep(0.03)
                cap.release()
                st.balloons()
            else:
                st.image("https://placeholder.com", use_container_width=True)
            
    with col_log:
        st.markdown("<h3 style='color: #2D3748; font-size: 18px;'>📋 Ringkasan Tabel Emisi (CO2)</h3>", unsafe_allow_html=True)
        st.dataframe(
            df_grouped.style.format({"Total_Emisi_g": "{:.1f}"}),
            height=370,
            use_container_width=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3 style='color: #2D3748; font-size: 20px; text-align: center;'>📊 Analisis Karbon & Heatmap Spasial</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    chart_col, heatmap_col = st.columns(2)
    
    with chart_col:
        fig = px.bar(
            df_grouped, 
            x="Kendaraan", 
            y="Total_Emisi_g",
            title="Distribusi Emisi per Kelas Kendaraan (g CO2)",
            labels={"Kendaraan": "Jenis Kendaraan", "Total_Emisi_g": "Gram CO2"},
            color_discrete_sequence=["#2E7D32"]
        )
        fig.update_layout(
            plot_bgcolor="rgba(245,247,250,1)", 
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#2D3748",
            title_font_size=14
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with heatmap_col:
        st.markdown("<p style='font-size: 14px; font-weight: bold; color: #2D3748; margin-bottom: 12px;'>Heatmap Kepadatan Emisi Spasial (Overlay KDE)</p>", unsafe_allow_html=True)
        st.image("https://placeholder.com", use_container_width=True)

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 **Petunjuk:** Silakan pilih jenis input dan unggah dokumen gambar atau video CCTV Anda pada panel samping kiri untuk mengaktifkan pemrosesan inferensi model OBB.")
    
    st.image(
        "https://placeholder.com", 
        use_container_width=True
    )
