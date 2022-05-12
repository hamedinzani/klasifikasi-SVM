from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve

# memanggil model
loaded_model = pickle.load(
    open('D:\Kuliah\TUGAS AKHIR\Program\modelsvm5.sav', 'rb'))


def prediksikonsentrasi(baru):
    # baru = [[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,0, 0, 0, 0, 1, 2, 1, 0, 1, 1, 1, 0, 0, 0, 2]]
    hasilPrediksi = loaded_model.predict(baru)
    if (hasilPrediksi == 0):
        return'WEM'
    if (hasilPrediksi == 1):
        return'SCR'
    if (hasilPrediksi == 2):
        return'KKO'


st.title(
    'Klasifikasi Konsentrasi Jurusan Informatika Menggunakan Algoritma SVM')

df = pd.read_csv(
    "D:\Kuliah\TUGAS AKHIR\Program\datasetnilaiakademik.csv")

if st.sidebar.checkbox('Tampilkan Dataset'):
    st.subheader('Dataset')
    st.write(df)


df['inggrisa'] = df['inggrisa'].str.lower()
df['alproa'] = df['alproa'].str.lower()
df['aljabarlinear'] = df['aljabarlinear'].str.lower()
df['fisikadasar'] = df['fisikadasar'].str.lower()
df['kalkulusa'] = df['kalkulusa'].str.lower()
df['aptia'] = df['aptia'].str.lower()
df['inggrisb'] = df['inggrisb'].str.lower()
df['alprob'] = df['alprob'].str.lower()
df['kalkulusb'] = df['kalkulusb'].str.lower()
df['aptib'] = df['aptib'].str.lower()
df['statistika'] = df['statistika'].str.lower()
df['pemdas'] = df['pemdas'].str.lower()
df['pemdasprak'] = df['pemdasprak'].str.lower()
df['pemweb'] = df['pemweb'].str.lower()
df['pemwebprak'] = df['pemwebprak'].str.lower()
df['pbo'] = df['pbo'].str.lower()
df['pboprak'] = df['pboprak'].str.lower()
df['sbd'] = df['sbd'].str.lower()
df['sbdprak'] = df['sbdprak'].str.lower()
df['sd'] = df['sd'].str.lower()
df['sdprak'] = df['sdprak'].str.lower()
df['jarkom'] = df['jarkom'].str.lower()
df['jarkomprak'] = df['jarkomprak'].str.lower()
df['matdis'] = df['matdis'].str.lower()
df['mpa'] = df['mpa'].str.lower()
df['sbdl'] = df['sbdl'].str.lower()
df['sbdlprak'] = df['sbdlprak'].str.lower()
df['sc'] = df['sc'].str.lower()
df['tbo'] = df['tbo'].str.lower()
if st.sidebar.checkbox('Ubah Ke huruf kecil'):
    st.subheader('Dataset Menjadi Huruf Kecil')
    st.write(df)


# mengisi nilai kosong
df['alprob'] = df['alprob'].fillna("c")
df['pemdas'] = df['pemdas'].fillna("b")
df['pemdasprak'] = df['pemdasprak'].fillna("b")
df['pemweb'] = df['pemweb'].fillna("b")
df['pemwebprak'] = df['pemwebprak'].fillna("b")
df['pbo'] = df['pbo'].fillna("b")
df['pboprak'] = df['pboprak'].fillna("b")
df['sbd'] = df['sbd'].fillna("b")
df['sbdprak'] = df['sbdprak'].fillna("b")
df['sd'] = df['sd'].fillna("b")
df['sdprak'] = df['sdprak'].fillna("b")
df['jarkom'] = df['jarkom'].fillna("b")
df['jarkomprak'] = df['jarkomprak'].fillna("b")
df['matdis'] = df['matdis'].fillna("b")
df['mpa'] = df['mpa'].fillna("b")
df['sbdl'] = df['sbdl'].fillna("b")
df['sbdlprak'] = df['sbdlprak'].fillna("b")
df['sc'] = df['sc'].fillna("b")
df['tbo'] = df['tbo'].fillna("c")
if st.sidebar.checkbox('Isi Nilai Kosong'):
    st.subheader('Nilai Kosong Data Set Terisi')
    st.write(df)

# merubah huruf jadi angka
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['inggrisa'] = labelencoder.fit_transform(df['inggrisa'])
df['alproa'] = labelencoder.fit_transform(df['alproa'])
df['aljabarlinear'] = labelencoder.fit_transform(df['aljabarlinear'])
df['fisikadasar'] = labelencoder.fit_transform(df['fisikadasar'])
df['kalkulusa'] = labelencoder.fit_transform(df['kalkulusa'])
df['aptia'] = labelencoder.fit_transform(df['aptia'])
df['inggrisb'] = labelencoder.fit_transform(df['inggrisb'])
df['alprob'] = labelencoder.fit_transform(df['alprob'])
df['kalkulusb'] = labelencoder.fit_transform(df['kalkulusb'])
df['aptib'] = labelencoder.fit_transform(df['aptib'])
df['statistika'] = labelencoder.fit_transform(df['statistika'])
df['pemdas'] = labelencoder.fit_transform(df['pemdas'])
df['pemdasprak'] = labelencoder.fit_transform(df['pemdasprak'])
df['pemweb'] = labelencoder.fit_transform(df['pemweb'])
df['pemwebprak'] = labelencoder.fit_transform(df['pemwebprak'])
df['pbo'] = labelencoder.fit_transform(df['pbo'])
df['pboprak'] = labelencoder.fit_transform(df['pboprak'])
df['sbd'] = labelencoder.fit_transform(df['sbd'])
df['sbdprak'] = labelencoder.fit_transform(df['sbdprak'])
df['sd'] = labelencoder.fit_transform(df['sd'])
df['sdprak'] = labelencoder.fit_transform(df['sdprak'])
df['jarkom'] = labelencoder.fit_transform(df['jarkom'])
df['jarkomprak'] = labelencoder.fit_transform(df['jarkomprak'])
df['matdis'] = labelencoder.fit_transform(df['matdis'])
df['mpa'] = labelencoder.fit_transform(df['mpa'])
df['sbdl'] = labelencoder.fit_transform(df['sbdl'])
df['sbdlprak'] = labelencoder.fit_transform(df['sbdlprak'])
df['sc'] = labelencoder.fit_transform(df['sc'])
df['tbo'] = labelencoder.fit_transform(df['tbo'])

if st.sidebar.checkbox('Ubah Huruf Menjadi Angka'):
    st.subheader('Dataset Menjadi Huruf Angka')
    st.write(df)

# feature engineering (mengambil atribut yang ingin digunakan)
X = df[['inggrisa', 'alproa', 'aljabarlinear', 'fisikadasar', 'kalkulusa', 'aptia', 'inggrisb',
        'alprob', 'kalkulusb', 'aptib', 'statistika', 'pemdas', 'pemdasprak', 'pemweb', 'pemwebprak',
        'pbo', 'pboprak', 'sbd', 'sd', 'sdprak', 'sbdprak', 'jarkom', 'jarkomprak', 'matdis', 'mpa', 'sbdl', 'sbdlprak', 'sc', 'tbo']]

# memisahkan Feature dengan class
y = df.konsentrasi

# membagi dari total data menjadi 70% data Latih dan 30% data Uji
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# if st.sidebar.button('Membagi Dataset menjadi Data Train dan Test'):
#     st.subheader('DataTrain Fitur dan Target')
#     st.write(X_train, y_train)
#     st.subheader('DataTest Fitur dan Target')
#     st.write(X_test, y_test)

# mencaritahu proporsi unik di kolom atribut
a = df.konsentrasi.value_counts()
if st.sidebar.checkbox('proporsi unik kolom atribut'):
    st.write(a)

# if st.sidebar.checkbox('proporsi unik kolom'):
#     st.title('Membagi data train dan test')
#     number = st.number_input('Masukan proporsi (contoh : 0.1 berarti 10%)')
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number)
#     st.subheader('DataTrain Fitur dan Target')
#     st.write(X_train, y_train)
#     st.subheader('DataTest Fitur dan Target')
#     st.write(X_test, y_test)

clf = svm.SVC(kernel='linear', probability=True)


number = st.sidebar.number_input(
    'Masukan proporsi (contoh : 0.1 berarti 10%)', min_value=0.1, value=0.30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number)


if st.sidebar.checkbox('proporsi'):
    st.title('Membagi data train dan test')
    st.subheader('DataTrain Fitur dan Target')
    st.write(X_train, y_train)
    st.subheader('DataTest Fitur dan Target')
    st.write(X_test, y_test)


clf = clf.fit(X_train, y_train)


# mendefinisikan model svm
# clf = svm.SVC(kernel='linear', probability=True)
# if st.button('Mendefinisikan Model SVM'):
#     st.write('clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,))')
# melakukan training model
# clf = clf.fit(X_train, y_train)
# if st.button('Melakukan Training Model SVM'):
#     st.write('clf = clf.fit(X_train, y_train)')

# cv


if st.sidebar.checkbox('Melakukan Pelatihan Model'):
    st.write('Bobot W', clf.coef_)
    st.write('Bias B ', clf.intercept_)
    st.write('Hasil Pelatihan', clf.predict(X_train))
    scores = cross_val_score(clf, X_train, y_train, cv=20)
    st.write(scores)


# prediksi data uji


hasilPrediksi = clf.predict(X_test)
if st.sidebar.checkbox('Prediksi Data Test'):
    st.write('Label Sebenarbya', y_test)
    st.write('Hasil Prediksi', hasilPrediksi)

proba = clf.predict_proba(X_test)
# membuat hasil prediksi dari array menjadi dataFrame
# hasil = pd.DataFrame(hasilPrediksi, columns=['Predicted'])
# st.write(hasil)

# membuat confusion matrix
cm = confusion_matrix(y_test, hasilPrediksi)

# mendefinisikan index confusion matrix
cm_df = pd.DataFrame(cm,
                     index=['WEM', 'SCR', 'KKO'],
                     columns=['WEM', 'SCR', 'KKO'])

# Plotting the confusion matrix
if st.sidebar.button("Show Correlation Plot"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("### Heatmap")
    fig, ax = plt.subplots(figsize=(5, 4))
    st.write(sns.heatmap(cm_df, annot=True, fmt='d'))
    st.pyplot()
    # model report
    target = ['WEM', 'SCR', 'KKO']
    st.write(classification_report(
        y_test, hasilPrediksi, target_names=target))


if st.sidebar.checkbox("MSE"):
    st.write('Mean Squared Error')
    st.write(metrics.mean_squared_error(y_test, hasilPrediksi))

    # st.write(cm_df)

if st.sidebar.checkbox("ROC"):
    st.write("ROC")
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 3

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, proba[:, i], pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--',
             color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--',
             color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--',
             color='blue', label='Class 2 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC', dpi=300)
    st.pyplot()


def main():
    # membuat form input data baru

    inggrisa = st.text_input('Nilai Bahasa Inggris A')
    alproa = st.text_input('Nilai Algoritma pemrograman I')
    aljabarlinear = st.text_input('Nilai Aljabar Linear dan Matrix')
    fisikadasar = st.text_input('Nilai Fisika Dasar')
    kalkulusa = st.text_input('Nilai kalkulus I ')
    aptia = st.text_input('Nilai Apti I ')
    inggrisb = st.text_input('Nilai Bahasa Inggris B')
    alprob = st.text_input('Nilai Algoritma Pemrograman II')
    kalkulusb = st.text_input('Nilai Kalkulus II ')
    aptib = st.text_input('Nilai Apti II ')
    statistika = st.text_input('Nilai Statistika')
    pemdas = st.text_input('Nilai Pemrograman Dasar')
    pemdasprak = st.text_input('Nilai Pemdas Praktik ')
    pemweb = st.text_input('Nilai Pemrograman WEB')
    pemwebprak = st.text_input('Nilai PemWEB Praktik')
    pbo = st.text_input('Nilai Pemrograman Berorientasi Objek')
    pboprak = st.text_input('Nilai PBO Praktik ')
    sbd = st.text_input('Nilai Sistem Basis Data')
    sd = st.text_input('Nilai Struktur Data')
    sdprak = st.text_input('Nilai Struktur Data Praktik')
    sbdprak = st.text_input('Nilai SBD Praktik')
    jarkom = st.text_input('Nilai Jaringan Komputer')
    jarkomprak = st.text_input('Nilai Jarkom Praktik ')
    matdis = st.text_input('Nilai Matematika Diskrit')
    mpa = st.text_input('Nilai Manajemen Penggembangan Aplikasi')
    sbdl = st.text_input('Nilai Sistem Basis Data Lanjut')
    sbdlprak = st.text_input('Nilai SBDL Praktik')
    sc = st.text_input('Nilai Sistem Cerdas')
    tbo = st.text_input('Nilai Teori Bahasa Automata')

    data = pd.DataFrame({'inggrisa': [inggrisa, alproa, aljabarlinear, fisikadasar, kalkulusa, aptia, inggrisb, alprob, kalkulusb, aptib, statistika,
                                      pemdas, pemdasprak, pemweb, pemwebprak, pbo, pboprak, sbd, sd, sdprak, sbdprak, jarkom, jarkomprak, matdis, mpa, sbdl, sbdlprak, sc, tbo],
                         })

    all_labelencoders = {}
    cols = ['inggrisa']

    for name in cols:
        labelencoder = LabelEncoder()
        all_labelencoders[name] = labelencoder

        labelencoder.fit(data[name])
        data[name] = labelencoder.transform(data[name])
        data_tr = data.transpose()

        # prediksi
    prediksi = ''

    # membuat  button untuk prediksi
    if st.button('Prediksi Konsentrasi'):
        prediksi = prediksikonsentrasi(data_tr)
        prediction = clf.predict(data_tr)
        output = int(prediction[0])
        probas = clf.predict_proba(data_tr)
        output_probability = float(probas[:, output].round(3))
        result = {"confidence_score": output_probability}
        prediksi = prediksikonsentrasi(data_tr)
        st.success(prediksi)

        st.success(result)


if st.sidebar.checkbox('Prediksi Data Baru'):
    st.title('Inputkan Nilai Akademik Untuk Melakukan Klasifikasi')
    st.write(main())
