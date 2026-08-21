import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import os

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

uploaded_file = st.sidebar.file_uploader(
    "Unggah Foto Udara Lintasan Jalan", 
    type=["jpg", "jpeg", "png"]
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #4A5568; font-size: 14px; margin-bottom: 5px;'>🚀 AI CONFIGURATION</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<span style='background-color: #E8F5E9; color: #2E7D32; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold;'>YOLOv8-OBB Active</span>", unsafe_allow_html=True)

MODEL_PATH = "models/best.pt"
if os.path.exists(MODEL_PATH):
    st.sidebar.markdown(f"<p style='font-size: 12px; color: #2E7D32; margin-top: 10px;'>✔️ Model Loaded: <code>{MODEL_PATH}</code></p>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"<p style='font-size: 12px; color: #E53E3E; margin-top: 10px;'>❌ Model tidak ditemukan di <code>{MODEL_PATH}</code></p>", unsafe_allow_html=True)

EMISSION_FACTORS = {
    'bike': 115,
    'car': 192,
    'taxi': 192,
    'bus': 822,
    'truck': 900,
    'other_vehicle': 200
}
road_length_km = 0.1

if uploaded_file is not None and os.path.exists(MODEL_PATH):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    
    with st.spinner("Model OBB sedang mengekstrak koordinat spasial kendaraan..."):
        model = YOLO(MODEL_PATH)
        results = model(opencv_img)
        result = results[0]
        
        annotated_img = result.plot()
        orig_img = result.orig_img
        
        if result.obb is not None:
            detected_classes = [model.names[int(cls_id)] for cls_id in result.obb.cls]
            centers_x = [box[0].item() for box in result.obb.xywhr]
            centers_y = [box[1].item() for box in result.obb.xywhr]
        else:
            detected_classes = []
            centers_x = []
            centers_y = []
            
    total_co2 = 0
    emission_details = []
    for vehicle in detected_classes:
        factor = EMISSION_FACTORS.get(vehicle, 150)
        co2_emitted = factor * road_length_km
        total_co2 += co2_emitted
        emission_details.append({'Kendaraan': vehicle, 'Emisi (g CO2)': co2_emitted})
        
    df_emissions = pd.DataFrame(emission_details)
    
    if not df_emissions.empty:
        df_grouped = df_emissions.groupby('Kendaraan').agg(
            Jumlah_Kendaraan=('Kendaraan', 'count'),
            Total_Emisi_g=('Emisi (g CO2)', 'sum')
        ).reset_index()
    else:
        df_grouped = pd.DataFrame(columns=['Kendaraan', 'Jumlah_Kendaraan', 'Total_Emisi_g'])

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Total Objek Terdeteksi", value=f"{len(df_emissions)} Unit")
    kpi2.metric(label="Total Akumulasi Emisi", value=f"{total_co2:.2f} g CO2")
    kpi3.metric(label="Metode Pemetaan", value="Oriented Bounding Box (OBB)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_vid, col_log = st.columns(2)
    
    with col_vid:
        st.markdown("<h3 style='color: #2D3748; font-size: 18px;'>🎥 Hasil Deteksi YOLOv8-OBB</h3>", unsafe_allow_html=True)
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Arah Kotak Deteksi Mengikuti Sudut Kemiringan Objek", use_container_width=True)
        st.success("Analisis Citra Spasial Selesai!")
            
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
        if not df_grouped.empty:
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
        else:
            st.info("Tidak ada data untuk merender grafik.")
        
    with heatmap_col:
        st.markdown("<p style='font-size: 14px; font-weight: bold; color: #2D3748; margin-bottom: 12px;'>Heatmap Kepadatan Emisi Spasial (KDE Plot Overlay)</p>", unsafe_allow_html=True)
        
        if centers_x and centers_y:
            fig_hp, ax = plt.subplots(figsize=(6, 3.7))
            orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            ax.imshow(orig_rgb)
            
            sns.kdeplot(
                x=centers_x, 
                y=centers_y, 
                cmap="Reds", 
                fill=True, 
                bw_adjust=0.5, 
                alpha=0.5, 
                ax=ax
            )
            ax.set_xlim(0, result.orig_shape[1])
            ax.set_ylim(result.orig_shape[0], 0)
            ax.axis('off')
            
            st.pyplot(fig_hp, use_container_width=True)
            plt.close(fig_hp)
        else:
            st.info("Heatmap tidak dapat ditampilkan karena tidak ada kendaraan yang terdeteksi.")

elif not os.path.exists(MODEL_PATH):
    st.error(f"Aplikasi dihentikan. File model tidak ditemukan pada direktori `{MODEL_PATH}`. Harap unggah bobot model Anda terlebih dahulu ke folder proyek.")
else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 **Petunjuk:** Silakan unggah dokumen gambar foto udara Anda pada panel samping kiri untuk mengeksekusi inferensi model YOLOv8-OBB dan memetakan kepadatan emisi.")

