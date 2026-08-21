import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="ESG Drone Carbon Detection",
    page_icon="🌱",
    layout="wide"
)

# Tema Hijau Putih via Custom CSS
st.markdown("""
    <style>
    :root {
        --primary-color: #1E5631;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    h1, h2, h3, p {
        color: #1E5631 !important;
    }
    .sidebar .sidebar-content {
        background-color: #F4F9F4 !important;
    }
    .metric-box {
        background-color: #F4F9F4;
        border: 2px solid #1E5631;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .metric-val {
        font-size: 32px;
        font-weight: bold;
        color: #1E5631;
    }
    .metric-lbl {
        font-size: 16px;
        color: #4C7A53;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Memuat Model YOLOv8 OBB (Mencari file best.pt di direktori aktif)
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"❌ File model '{model_path}' tidak ditemukan di folder aplikasi Anda. Pastikan file sudah dipindahkan!")
        return None

model = load_model()

# Faktor Emisi CO2 Berdasarkan Notebook (g CO2 / km)
EMISSION_FACTORS = {
    'bike': 115,
    'car': 192,
    'taxi': 192,
    'bus': 822,
    'truck': 900,
    'other_vehicle': 200
}
road_length_km = 0.1

# --- SIDEBAR ---
st.sidebar.header("🌱 MENU UTAMA")

# 1. Bagian Unggah Foto Drone
uploaded_file = st.sidebar.file_uploader(
    "Unggah Foto Udara Drone", 
    type=['jpg', 'jpeg', 'png']
)

st.sidebar.markdown("---")

# 2. Bagian Keunggulan Fitur
st.sidebar.subheader("✨ Apa Bagusnya Aplikasi Ini?")
st.sidebar.markdown("""
* **Deteksi Presisi (OBB):** Menggunakan kotak miring (*Oriented Bounding Box*) yang sangat akurat untuk mendeteksi kendaraan dari sudut pandang udara lurus (drone).
* **Audit ESG Instan:** Mengonversi objek kendaraan langsung menjadi metrik estimasi emisi karbon secara *real-time*.
* **Visualisasi Spasial:** Dilengkapi peta panas (*heatmap*) untuk mendeteksi titik jenuh polusi udara di jalan raya.
""")

# --- HALAMAN UTAMA ---
st.title("🌱 ESG Drone Carbon Detection Dashboard")
st.write("Analisis jejak karbon kendaraan bermotor secara otomatis menggunakan citra udara drone.")

if uploaded_file is not None and model is not None:
    # Konversi file unggahan ke format OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Proses Prediksi Model YOLO
    results = model(orig_img)
    result = results[0]
    
    # Ekstraksi Data Hasil Deteksi
    detected_classes = []
    centers_x = []
    centers_y = []
    
    if result.obb is not None:
        detected_classes = [model.names[int(cls_id)] for cls_id in result.obb.cls]
        centers_x = [box[0].item() for box in result.obb.xywhr]
        centers_y = [box[1].item() for box in result.obb.xywhr]

    # Hitung Kalkulasi Emisi dan Kelompokkan Data
    total_vehicles = len(detected_classes)
    total_co2 = 0
    emission_details = []
    
    for vehicle in detected_classes:
        factor = EMISSION_FACTORS.get(vehicle, 150)
        co2_emitted = factor * road_length_km
        total_co2 += co2_emitted
        emission_details.append({'Kendaraan': vehicle, 'Emisi (g CO2)': co2_emitted})
        
    df_emissions = pd.DataFrame(emission_details)
    if not df_emissions.empty:
        df_grouped = df_emissions.groupby('Kendaraan').sum().reset_index()
    else:
        df_grouped = pd.DataFrame(columns=['Kendaraan', 'Emisi (g CO2)'])

    # --- KOTAK RINGKASAN METRIK (KPI CARDS) ---
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">TOTAL KENDARAAN TERDETEKSI</div>
                <div class="metric-val">{total_vehicles} Unit</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">ESTIMASI TOTAL EMISI KARBON</div>
                <div class="metric-val">{total_co2:.2f} g CO₂</div>
            </div>
        """, unsafe_allow_html=True)

    # Buat Gambar yang Ber-annotated (Box Deteksi)
    annotated_img = result.plot()
    text_str = f"Total Emisi: {total_co2:.2f} g CO2"
    cv2.putText(annotated_img, text_str, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 5, cv2.LINE_AA)

    # --- GRID VISUALISASI ---
    st.subheader("📊 Hasil Analisis Visual & Spasial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📷 Foto Asli Drone")
        st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.markdown("### 🎯 Deteksi Berkotak OBB")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🔥 Heatmap Kepadatan Emisi")
        fig_heat, ax_heat = plt.subplots(figsize=(10, 8))
        ax_heat.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        if centers_x and centers_y:
            sns.kdeplot(
                x=centers_x, y=centers_y, 
                cmap="Greens", fill=True, 
                bw_adjust=0.5, alpha=0.6, ax=ax_heat
            )
        ax_heat.set_xlim(0, result.orig_shape[1])
        ax_heat.set_ylim(result.orig_shape[0], 0)
        ax_heat.axis('off')
        st.pyplot(fig_heat)
        
    with col4:
        st.markdown("### 📈 Distribusi Emisi per Kelas")
        if not df_grouped.empty:
            fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
            sns.barplot(
                data=df_grouped, x='Kendaraan', y='Emisi (g CO2)', 
                ax=ax_bar, palette='Greens_r'
            )
            ax_bar.set_ylabel("Gram CO2", fontsize=12)
            ax_bar.set_xlabel("Jenis Kendaraan", fontsize=12)
            # Modifikasi estetika grafik biar menyatu dengan tema
            fig_bar.patch.set_facecolor('#FFFFFF')
            ax_bar.set_facecolor('#F4F9F4')
            st.pyplot(fig_bar)
        else:
            st.info("Tidak ada data emisi untuk membuat grafik batang.")

    # --- TABEL RINCIAN ---
    st.markdown("### 📋 Tabel Rincian Emisi CO2")
    if not df_grouped.empty:
        st.dataframe(df_grouped.style.format({'Emisi (g CO2)': '{:.2f}'}), use_container_width=True)
    else:
        st.info("Belum ada kendaraan yang terdeteksi.")

else:
    st.info("💡 Silakan unggah foto udara drone pada sidebar sebelah kiri untuk memulai analisis deteksi emisi karbon.")
