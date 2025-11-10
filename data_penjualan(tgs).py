import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

# Judul aplikasi
st.title("📊 Analisis Data Penjualan dan Prediksi Jenis Produk")

# 1. Membaca dataset dari file lokal
try:
    data = pd.read_csv("data_penjualan.csv", sep=';')
    st.success("✅ Dataset berhasil dibaca dari file lokal!")
except FileNotFoundError:
    st.error("❌ File 'data_penjualan.csv' tidak ditemukan di folder ini.")
    st.stop()

# 2. Tampilkan data awal
st.subheader("🧾 Data Awal")
st.dataframe(data.head(10))
st.write("**Jumlah Data:**", len(data))
st.write("**Kolom Dataset:**", list(data.columns))

# 3. Pisahkan fitur dan target
X = data.drop(columns=['Jenis Produk', 'Tanggal'])
y = data['Jenis Produk']

# 4. Tangani missing value
X = X.fillna(0)

# 5. Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 7. Buat dan latih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Prediksi pada data uji
y_pred = model.predict(X_test)

# 9. Evaluasi model
acc = accuracy_score(y_test, y_pred)
st.subheader("🎯 Evaluasi Model")
st.write(f"**Akurasi Model:** {acc:.2f}")

# 10. Laporan klasifikasi
labels_used = unique_labels(y_test, y_pred)
report = classification_report(
    y_test, y_pred,
    labels=labels_used,
    target_names=le.classes_[labels_used],
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# 11. Tampilkan hasil prediksi
hasil_prediksi = X_test.copy()
hasil_prediksi["Jenis Produk Asli"] = le.inverse_transform(y_test)
hasil_prediksi["Jenis Produk Prediksi"] = le.inverse_transform(y_pred)

st.subheader("📋 Hasil Prediksi Data Uji")
st.dataframe(hasil_prediksi.head(10))

# 12. Tombol download hasil prediksi
csv = hasil_prediksi.to_csv(index=False).encode('utf-8')
st.download_button(
    label="💾 Download Hasil Prediksi CSV",
    data=csv,
    file_name='hasil_prediksi_data_penjualan.csv',
    mime='text/csv',
)

# 13. Input data baru untuk prediksi manual
st.subheader("🧮 Coba Prediksi Data Baru")

with st.form("form_prediksi"):
    jumlah_order = st.number_input("Jumlah Order", min_value=0, value=1000, step=100)
    harga = st.number_input("Harga Satuan", min_value=0, value=1500, step=50)
    total = st.number_input("Total Harga (Jumlah × Harga)", min_value=0, value=1500000, step=10000)
    
    submit = st.form_submit_button("Prediksi Jenis Produk")

if submit:
    # Buat DataFrame dari input user
    input_data = pd.DataFrame({
        "Jumlah Order": [jumlah_order],
        "Harga": [harga],
        "Total": [total]
    })
    
    # Prediksi
    pred = model.predict(input_data)
    pred_label = le.inverse_transform(pred)[0]
    
    st.success(f"🔮 Prediksi Jenis Produk: **{pred_label}**")

    st.info("Model memprediksi jenis produk berdasarkan pola data penjualan sebelumnya.")
