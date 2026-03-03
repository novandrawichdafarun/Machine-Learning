import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#! Url Dataset Telco Customer Churn 
data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

#? Memuat dataset ke dalam pandas DataFrame (Tabel)
print("Sedang mengunduh dan memuat data...\n")
df = pd.read_csv(data_url)

#? Menampilkan bentuk asli data
# print("\n=== bentuk asli Dataset ===")
# print(df.head())

#? Memeriksa informasi dataset (tipe data dan jumlah kolom/baris)
# print("\n=== Informasi Dataset ===")
# df.info()

#? Memerikasa jumlah data yang kosong (missing values)
# print("\n=== Jumlah Data yang kosong per Kolom ===")
# print(df.isnull().sum())

#? Mengubah TotalCharges menjadi angka numerik (float)
df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors='coerce')

#? Memeriksa berapa banyak data yang sekarang menjadi kosong (NaN)
# print("Jumlah data kosong pada TotalCharges setelah konversi:", df["TotalCharges"].isnull().sum())

#? Menangani data yang kosong
df.dropna(subset=['TotalCharges'], inplace=True)

#! Verifikasi tipe data
# print("\nTIpe data TotalCharges sekarang:", df['TotalCharges'].dtype)
# print("Ukuran dataset sekarang (baris, kolom):", df.shape)

#? Menghapus kolom yang tidak relevan
if 'customerID'in df.columns:
  df = df.drop('customerID', axis=1)
  
#? Memisahkan Fitur X dan target Y
x = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0}) #! Klasifikasi Yes/No -> 1/0

#? Membagi data menjadi Training Set (80%) dan Testing Set (20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#? Mendefinisikan kolom numerik dan kategorikal
categorical_cols = x_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = x_train.select_dtypes(include=['number']).columns.tolist()

#? Membuat Pipeline Transformasi (ColumnsTransformer)
preprocessor = ColumnTransformer(
  transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
  ]
)

#? Menerapkan transformasi ke data latih (fit_transform) dan data uji (transform)
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

# print("--- Rekayasa Fitur Selesai! ---")
# print(f"Bentuk X_train sebelum diproses: {x_train.shape}")
# print(f"Bentuk X_train setelah diproses: {x_train_processed.shape}")

#! Inisialisasi Model
log_reg = LogisticRegression(random_state=42, max_iter=1000)  # Baseline
rf_model = RandomForestClassifier(random_state=42, n_estimators=100) # 100 pohon/n_estimators

#? Melatih model (Train)
print("Sedang melatih model...")
log_reg.fit(x_train_processed, y_train)
rf_model.fit(x_train_processed, y_train)

#? Prediksi pada Data Uji (Test)
y_pred_log_reg = log_reg.predict(x_test_processed)
y_pred_rf = rf_model.predict(x_test_processed)

#? Evaluasi Model (Melihat Rapor)
# print("\n=== Rapor Evaluasi: Logistic Regression ===")
# print(classification_report(y_test, y_pred_log_reg))

# print("\n=== Rapor Evaluasi: Random Forest ===")
# print(classification_report(y_test, y_pred_rf))

#? Menyimpan preprocessor (aturan scaling & encoding) dan model pemenang (Logistic Regression)
joblib.dump(preprocessor, 'churn_preprocessor.pkl')
joblib.dump(log_reg, 'churn_model.pkl')
print("Model dan Preprocessor berhasil disimpan ke dalam file .pkl!\n")

#? Simulasi Produksi
loaded_preprocessor = joblib.load('churn_preprocessor.pkl')
loaded_model = joblib.load('churn_model.pkl')

pelanggan_baru = pd.DataFrame({
    'gender': ['Female'],
    'SeniorCitizen': [0],
    'Partner': ['Yes'],
    'Dependents': ['No'],
    'tenure': [2], #! Baru berlangganan 2 bulan
    'PhoneService': ['Yes'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['Yes'],
    'StreamingMovies': ['Yes'],
    'Contract': ['Month-to-month'], #! Kontrak bulanan biasanya rawan churn
    'PaperlessBilling': ['Yes'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [85.0], #! Tagihan cukup tinggi
    'TotalCharges': [170.0]
})

#! Prediksi
pelanggan_baru_processed = loaded_preprocessor.transform(pelanggan_baru)

prediksi = loaded_model.predict(pelanggan_baru_processed)
probabilitas = loaded_model.predict_proba(pelanggan_baru_processed)[0]

print("--- Hasil Prediksi Sistem ---")
if prediksi[0] == 1:
    print("⚠️ PERINGATAN: Pelanggan ini KEMUNGKINAN BESAR AKAN CHURN (Berhenti Berlangganan).")
else:
    print("✅ AMAN: Pelanggan ini kemungkinan besar akan BERTANHAN.")

print(f"Probabilitas Churn: {probabilitas[1]*100:.2f}%")