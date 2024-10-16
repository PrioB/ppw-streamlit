import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Fungsi untuk melakukan preprocessing teks
def preprocess_text(text):
    # Tambahkan logika preprocessing di sini sesuai kebutuhan
    return text.lower()

# Judul aplikasi
st.title('Klasifikasi Berita Otomotif dan Sport')

# Input teks untuk klasifikasi
input_text = st.text_area("Masukkan teks berita di sini:")

# Tombol untuk melatih model
if st.button('Latih Model'):
    # Contoh data (gantilah ini dengan data yang Anda ambil dari scraping)
    data = {
        'isi_berita': [
            "Berita otomotif terbaru tentang mobil baru",
            "Hasil pertandingan olahraga sepak bola",
            "Inovasi di dunia kendaraan listrik",
            "Update tentang tim basket nasional"
        ],
        'kategori': ['Otomotif', 'Sport', 'Otomotif', 'Sport']
    }

    df = pd.DataFrame(data)

    # Preprocessing dan membangun model
    df['isi_berita_clean'] = df['isi_berita'].apply(preprocess_text)
    
    # Membuat pipeline
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipeline.fit(df['isi_berita_clean'], df['kategori'])
    
    # Menyimpan model
    joblib.dump(pipeline, 'logistic_model.joblib')
    st.success('Model berhasil dilatih dan disimpan!')

# Tombol untuk melakukan klasifikasi
if st.button('Klasifikasikan'):
    if input_text:
        try:
            model = joblib.load('logistic_model.joblib')
            processed_text = preprocess_text(input_text)
            prediction = model.predict([processed_text])[0]
            st.write(f'Kategori: {prediction}')
        except FileNotFoundError:
            st.error('Model belum dilatih atau tidak ditemukan. Silakan latih model terlebih dahulu.')
    else:
        st.warning('Masukkan teks berita untuk klasifikasi.')

# Tombol untuk memuat model (opsional)
if st.button('Muatan Model'):
    try:
        model = joblib.load('logistic_model.joblib')
        st.success('Model berhasil dimuat!')
    except FileNotFoundError:
        st.error('Model tidak ditemukan.')
