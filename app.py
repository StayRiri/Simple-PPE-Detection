import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import time
import os
import tempfile
from ultralytics import YOLO

MODEL_PATH = 'best.pt'
LOG_FILE = 'log_pelanggaran.csv'

# Konfigurasi Halaman Web
st.set_page_config(page_title="Monitor K3 AI", page_icon="🚧", layout="wide")
st.title("Dashboard Monitoring K3 Konstruksi")
st.markdown("Sistem pendeteksi kelengkapan Alat Pelingdung Diri.")

# Buat file CSV jika belum ada (Dengan kolom ID Pekerja)
if not os.path.exists(LOG_FILE):
    df_awal = pd.DataFrame(columns=["Waktu", "ID Pekerja", "Jenis Pelanggaran", "Lokasi"])
    df_awal.to_csv(LOG_FILE, index=False)

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("Pengaturan Sumber Video")
sumber_opsi = st.sidebar.radio("Pilih Sumber Input:", ["Webcam", "Upload Video"])

file_video = None
if sumber_opsi == "Upload Video":
    file_video = st.sidebar.file_uploader("Upload video (.mp4, .avi, .mov)", type=['mp4', 'avi', 'mov'])

# --- TATA LETAK UTAMA ---
col1, col2 = st.columns([6, 4])

with col1:
    st.subheader("Visualisasi Deteksi Alat Pelindung Diri")
    frame_placeholder = st.empty()
    start_button = st.button("▶️ Mulai Deteksi", type="primary")
    stop_button = st.button("⏹️ Hentikan")

with col2:
    st.subheader("📋 Log Pelanggaran")
    table_placeholder = st.empty()

def load_logs():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE).tail(12)
    return pd.DataFrame()

# Tampilkan tabel kosong/awal dengan rapi
table_placeholder.dataframe(load_logs(), use_container_width=True, hide_index=True)

# --- LOGIKA UTAMA DETEKSI ---
if start_button:
    try:
        if sumber_opsi == "Upload Video" and file_video is None:
            st.error("Silakan upload file video terlebih dahulu di sidebar kiri!")
        else:
            model = YOLO(MODEL_PATH)
            
            if sumber_opsi == "Webcam":
                cap = cv2.VideoCapture(0)
            else:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file_video.read())
                cap = cv2.VideoCapture(tfile.name)
            
            # MEMORI CERDAS & MAPPING ID
            pelanggar_tercatat = set() 
            pemetaan_id = {}         # Buku catatan untuk menerjemahkan ID YOLO ke ID berurutan
            id_urut_selanjutnya = 1  # Kita mulai berhitung dari 1
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.info("Pemutaran video selesai atau sumber terputus.")
                    break
                    
                frame = cv2.resize(frame, (640, 480))
                    
                # KUNCI UTAMA: Menggunakan fitur .track() bukan .predict()
                # persist=True artinya AI akan mengingat objek dari frame sebelumnya
                results = model.track(frame, persist=True, conf=0.5, verbose=False)
                
                boxes_person = []
                boxes_helmet = []
                boxes_vest = []
                boxes_gloves = []
                boxes_boots = []

                for result in results:
                    if result.boxes is None: continue
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = model.names[cls_id]
                        
                        # Ambil ID unik jika AI berhasil melacak objek tersebut
                        track_id = int(box.id[0]) if box.id is not None else -1

                        warna = (0, 255, 0) if class_name != 'Person' else (255, 0, 0)
                        
                        # --- MODIFIKASI MAPPING ID BERURUTAN ---
                        id_final = track_id
                        if class_name == 'Person' and track_id != -1:
                            # Jika ID dari YOLO belum ada di buku kita, catat dan beri nomor urut cantik
                            if track_id not in pemetaan_id:
                                pemetaan_id[track_id] = id_urut_selanjutnya
                                id_urut_selanjutnya += 1
                            
                            # Gunakan nomor urut cantik dari buku kita
                            id_final = pemetaan_id[track_id]
                            label_teks = f"Pekerja ID:{id_final}"
                        else:
                            label_teks = f"{class_name}"
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)
                        cv2.putText(frame, label_teks, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)

                        # Memisahkan hasil deteksi (PENTING: gunakan id_final di sini)
                        if class_name == 'Person': boxes_person.append((x1, y1, x2, y2, id_final))
                        elif class_name == 'Helmet': boxes_helmet.append((x1, y1, x2, y2))
                        elif class_name == 'Vest': boxes_vest.append((x1, y1, x2, y2))
                        elif class_name == 'Gloves': boxes_gloves.append((x1, y1, x2, y2))
                        elif class_name == 'Boots': boxes_boots.append((x1, y1, x2, y2))

                log_baru = []

                # --- VALIDASI PELANGGARAN ANTI-SPAM ---
                for px1, py1, px2, py2, track_id in boxes_person:
                    # Jika pekerja belum mendapat ID yang stabil dari AI, lewati dulu
                    if track_id == -1:
                        continue
                        
                    pakai_helm = any(hx1 >= px1 - 50 and hx2 <= px2 + 50 and hy1 >= py1 - 50 and hy2 <= py2 for hx1, hy1, hx2, hy2 in boxes_helmet)
                    pakai_rompi = any(vx1 >= px1 - 50 and vx2 <= px2 + 50 and vy1 >= py1 - 50 and vy2 <= py2 for vx1, vy1, vx2, vy2 in boxes_vest)
                    pakai_sarung = any(gx1 >= px1 - 50 and gx2 <= px2 + 50 and gy1 >= py1 - 50 and gy2 <= py2 for gx1, gy1, gx2, gy2 in boxes_gloves)
                    pakai_sepatu = any(bx1 >= px1 - 50 and bx2 <= px2 + 50 and by1 >= py1 - 50 and by2 <= py2 for bx1, by1, bx2, by2 in boxes_boots)

                    # Fungsi kecil untuk mencatat jika belum ada di memori
                    def catat_pelanggaran(kondisi_pakai, nama_pelanggaran):
                        if not kondisi_pakai:
                            kunci_memori = f"ID_{track_id}_{nama_pelanggaran}"
                            
                            # Jika kombinasi ID & Pelanggaran ini belum pernah dicatat
                            if kunci_memori not in pelanggar_tercatat:
                                log_baru.append({
                                    "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                    "ID Pekerja": f"Pekerja #{track_id}",
                                    "Jenis Pelanggaran": nama_pelanggaran, 
                                    "Lokasi": "Zona 1"
                                })
                                # Tambahkan ke memori 
                                pelanggar_tercatat.add(kunci_memori)

                    catat_pelanggaran(pakai_helm, "Tanpa Helm")
                    catat_pelanggaran(pakai_rompi, "Tanpa Rompi Safety")
                    catat_pelanggaran(pakai_sarung, "Tanpa Sarung Tangan")
                    catat_pelanggaran(pakai_sepatu, "Tanpa Sepatu Boots")

                # Jika ada pelanggaran baru dari ID yang baru, simpan & update layar
                if log_baru:
                    df_baru = pd.DataFrame(log_baru)
                    df_baru.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
                    table_placeholder.dataframe(load_logs(), use_container_width=True, hide_index=True)

                # Tampilkan frame video ke web
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                time.sleep(0.01)

            cap.release()
    
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")