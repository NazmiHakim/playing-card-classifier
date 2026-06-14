import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Klasifikasi Kartu Remi",
    page_icon="🃏",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    model_path = 'model_kartu_mobilenet.keras'
    if not os.path.exists(model_path):
        model_path = os.path.join('models', 'model_kartu_mobilenet.keras')
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di root maupun di folder 'models'!")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ['Club', 'Diamond', 'Heart', 'Spade']

# Antarmuka Pengguna
st.title("Prediksi Suit Kartu Remi")
st.markdown("""
Aplikasi web ini mendemonstrasikan hasil tugas UAS Machine Learning.
Model yang digunakan adalah **Deep Learning (Convolutional Neural Network)** dengan arsitektur **MobileNetV2**.

**Tingkat Akurasi Model: 82.5%**
""")

st.write("---")

uploaded_file = st.file_uploader("Unggah gambar kartu remi (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Gambar yang Anda unggah", width=300)
    
    st.write("🤖 Memproses gambar dengan MobileNetV2...")
    
    # Preprocessing citra
    img_tensor = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img_tensor = tf.image.resize(img_tensor, [224, 224])
    img_array = tf.expand_dims(img_tensor, 0)
    
    # Eksekusi prediksi
    if model is not None:
        predictions = model.predict(img_array)
        probabilities = predictions[0] 
        
        highest_prob_index = np.argmax(probabilities)
        predicted_class = class_names[highest_prob_index]
        confidence_score = probabilities[highest_prob_index] * 100
        
        # Tampilkan hasil prediksi
        st.write("---")
        if confidence_score >= 80:
            st.success(f"### Hasil Prediksi: **{predicted_class}**")
        elif confidence_score >= 50:
            st.warning(f"### Hasil Prediksi: **{predicted_class}** (Model kurang yakin)")
        else:
            st.error(f"### Hasil Prediksi: **{predicted_class}** (Sangat diragukan)")
            
        st.info(f"Tingkat Kepercayaan (Confidence): **{confidence_score:.2f}%**")
        
        # Detail probabilitas per kelas
        st.write("**Detail Probabilitas per Kelas:**")
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        st.bar_chart(prob_dict)
