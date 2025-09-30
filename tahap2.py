
import pandas as pd
import os

print("="*30)
print("TAHAP 1: INTEGRASI DATA")
print("="*30)

input_file = r"C:\Users\Agung\Downloads\Predicting Churn for Bank Customers\Churn_Modelling.csv"

# Memeriksa apakah file data awal ada
if not os.path.exists(input_file):
    print(f"❌ ERROR: File '{input_file}' tidak ditemukan.")
    print("Pastikan path file di atas sudah benar dan file tersebut ada.")
else:
    # Memuat dataset ke dalam DataFrame pandas
    df = pd.read_csv(input_file)
    print(f"✅ Dataset '{input_file}' berhasil dimuat.")
    print(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
    print("\nInfo awal dataset:")
    df.info()
    print("\nMenampilkan 5 baris pertama data:")
    print(df.head())

    # ==============================================================================
    # TAHAP 2: VALIDASI DATA
    # ==============================================================================
    print("\n" + "="*30)
    print("TAHAP 2: VALIDASI DATA")
    print("="*30)

    # 1. Pengecekan Missing Values
    print("\n1. Pengecekan Missing Values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("✅ Tidak ada missing values dalam dataset.")
    else:
        print("Jumlah missing values per kolom:")
        print(missing_values)

    # 2. Pengecekan Duplikasi
    print("\n2. Pengecekan Duplikasi:")
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows == 0:
        print("✅ Tidak ada baris data yang terduplikasi.")
    else:
        print(f"⚠️ Ditemukan {duplicate_rows} baris data duplikat.")
        # Opsi: df.drop_duplicates(inplace=True)

    # 3. Menghapus Kolom yang Tidak Relevan
    # Kolom seperti RowNumber, CustomerId, dan Surname tidak berguna untuk prediksi
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df_validated = df.drop(columns=columns_to_drop)
    print(f"\n3. Menghapus kolom tidak relevan: {', '.join(columns_to_drop)}")
    print(f"✅ Dataset sekarang memiliki {df_validated.shape[1]} kolom.")

    # 4. Menyimpan dataset yang sudah divalidasi
    # File ini akan disimpan di folder yang sama tempat Anda menjalankan script Python
    validated_file = 'data_validated.csv'
    df_validated.to_csv(validated_file, index=False)
    print(f"\n4. Dataset bersih telah disimpan sebagai '{validated_file}'")

    # ==============================================================================
    # TAHAP 3: ANALISIS DATA (EXPLORATORY DATA ANALYSIS)
    # ==============================================================================
    print("\n" + "="*30)
    print("TAHAP 3: ANALISIS DATA")
    print("="*30)

    # Menggunakan data yang sudah divalidasi (df_validated)
    
    # 1. Statistik Deskriptif untuk Kolom Numerik
    print("\n1. Statistik Deskriptif (Kolom Numerik):")
    # Menggunakan .T (transpose) agar lebih mudah dibaca
    print(df_validated.describe().T)

    # 2. Analisis Proporsi Churn (Target Variable 'Exited')
    print("\n2. Analisis Proporsi Churn (Nasabah Keluar):")
    # `normalize=True` untuk mendapatkan persentase
    churn_rate = df_validated['Exited'].value_counts(normalize=True) * 100
    print("0 = Tetap menjadi nasabah, 1 = Churn (keluar)")
    print(churn_rate)
    print(f"-> Sekitar {churn_rate.get(1, 0):.1f}% nasabah dalam dataset ini telah churn.")

    # 3. Analisis Churn berdasarkan Geografi
    print("\n3. Analisis Churn berdasarkan Geografi:")
    # Menggunakan groupby untuk mengelompokkan data berdasarkan negara
    # lalu menghitung rata-rata 'Exited' (karena Exited=1 untuk churn, rata-ratanya adalah churn rate)
    churn_by_geo = df_validated.groupby('Geography')['Exited'].mean() * 100
    print("Persentase Churn per Negara:")
    print(churn_by_geo.sort_values(ascending=False))
    print(f"-> Nasabah di {churn_by_geo.idxmax()} memiliki tingkat churn tertinggi.")

    # 4. Analisis Churn berdasarkan Gender
    print("\n4. Analisis Churn berdasarkan Gender:")
    churn_by_gender = df_validated.groupby('Gender')['Exited'].mean() * 100
    print("Persentase Churn per Gender:")
    print(churn_by_gender.sort_values(ascending=False))
    print(f"-> Nasabah {churn_by_gender.idxmax()} memiliki tingkat churn yang lebih tinggi.")

    # 5. Rata-rata Skor Kredit dan Saldo untuk Nasabah Churn vs. Non-Churn
    print("\n5. Analisis Fitur Numerik untuk Grup Churn vs Non-Churn:")
    churn_analysis_numeric = df_validated.groupby('Exited')[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']].mean()
    print(churn_analysis_numeric.T)
    print("\nAnalisis selesai.")