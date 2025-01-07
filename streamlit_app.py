import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import pickle

def main():
    st.set_page_config(page_title="Email Spam Classification Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    
    # Add dropdown for creators
    df = pd.read_csv('spambase_csv (1).csv')
    
    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()

    # # Unstack the correlation matrix and sort by correlation value
    # corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)

    # # Remove self-correlations
    # corr_pairs = corr_pairs[corr_pairs < 1]

    # # redundant_pairs = corr_pairs[corr_pairs > 0.95]
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    for column in df.columns:
        if outliers[column].sum() > 0:
            mean_value = df[column].mean()
            df.loc[outliers[column], column] = mean_value

    df.drop_duplicates(inplace=True)
    df.duplicated().sum()

    noisy_columns = df.var().sort_values(ascending=False).head(5)
    
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = int(len(df) * 0.01)  
    window_size = max(3, window_size)  
    for column in noisy_columns.index:
        smoothed_data = moving_average(df[column], window_size)
        df[column] = np.concatenate([smoothed_data, [np.nan] * (len(df[column]) - len(smoothed_data))])
    
    # for (col1, col2), value in redundant_pairs.items():
    #     if col1 in df.columns and col2 in df.columns:
    #         df.drop(col2, axis=1, inplace=True)

    # Temukan kolom yang memiliki korelasi di bawah ambang batas
    threshold = 0.5
    low_correlation_columns = set()

    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if abs(corr_matrix.loc[row, col]) < threshold and row != col:
                low_correlation_columns.add(col)

    # Hapus kolom yang memiliki korelasi rendah
    df.drop(columns=low_correlation_columns)
    
    # Fill NaN values with the mean of each column
    df.fillna(df.mean(), inplace=True)
    
    @st.cache_data
    def preprocess_data(df):
        scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        X = scaler.fit_transform(df.iloc[:, :-1])
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)

        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        return X_train_imputed, X_test_imputed, y_train, y_test, scaler, imputer,X_train, X_test

    @st.cache_data
    def train_logistic_regression(X_train_imputed, y_train):
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['saga', 'liblinear', 'lbfgs', 'newton-cg', 'sag'],
            'max_iter': [1000, 2000, 3000, 4000, 5000]
        }

        LogisticRegressionModel = LogisticRegression(random_state=33)
        grid_search = GridSearchCV(estimator=LogisticRegressionModel, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_imputed, y_train)

        return grid_search.best_estimator_

    @st.cache_data
    def train_kmeans(df):
        # Definisikan fitur untuk setiap cluster dalam dictionary
        clusters = {
        'cluster_1': ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 
                'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 
                'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 
                'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 
                'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 
                'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 
                'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 
                'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 
                'word_freq_data', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 
                'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 
                'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 
                'word_freq_edu', 'word_freq_table', 'capital_run_length_average'
                ],

        'cluster_2': ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 
                'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 
                'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 
                'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 
                'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 
                'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 
                'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 
                'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 
                'word_freq_data', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 
                'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 
                'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 
                'word_freq_edu', 'word_freq_table', 'char_freq_%3B', 'char_freq_%28', 
                'char_freq_%5B', 'char_freq_%21', 'char_freq_%24', 'char_freq_%23'],

        'cluster_3': ['capital_run_length_average','char_freq_%3B', 'char_freq_%28', 
                'char_freq_%5B', 'char_freq_%21', 'char_freq_%24', 'char_freq_%23',]
        }

        # Menghitung jumlah fitur untuk setiap cluster
        feature_sums = {cluster_name: df[features].sum().sum() for cluster_name, features in clusters.items()}

        # Menghitung total dari semua cluster
        total_sum = sum(feature_sums.values())

        # Probabilitas spam untuk setiap cluster
        spam_probabilities = {cluster_name: feature_sum / total_sum for cluster_name, feature_sum in feature_sums.items()}

        # Menambahkan kolom 'Cluster' ke DataFrame df
        df['cluster_1_feature_sum'] = df[clusters['cluster_1']].sum(axis=1)
        df['cluster_2_feature_sum'] = df[clusters['cluster_2']].sum(axis=1)
        df['cluster_3_feature_sum'] = df[clusters['cluster_3']].sum(axis=1)

        # Menetapkan cluster berdasarkan jumlah fitur
        def assign_cluster(row):
            if row['cluster_1_feature_sum'] > row['cluster_2_feature_sum'] and row['cluster_1_feature_sum'] > row['cluster_3_feature_sum']:
                return 1  # Cluster 1
            elif row['cluster_2_feature_sum'] > row['cluster_1_feature_sum'] and row['cluster_2_feature_sum'] > row['cluster_3_feature_sum']:
                return 2  # Cluster 2
            else:
                return 3  # Cluster 3

        # Terapkan fungsi untuk menetapkan cluster
        df['Cluster'] = df.apply(assign_cluster, axis=1)

        # Update spam probabilities for each cluster
        spam_probabilities = {
            'cluster_1': 0.2,
            'cluster_2': 0.7,
            'cluster_3': 0.75
        }

        # Fungsi untuk menghitung jumlah fitur dan menambahkan kolom spam probability
        def add_cluster_features(df, clusters, spam_probabilities):
            for cluster_name, features in clusters.items():
                df[f'{cluster_name}_feature_sum'] = df[features].sum(axis=1)
                cluster_index = int(cluster_name.split('_')[1]) - 1  # Mengambil index dari nama cluster
                df.loc[df['Cluster'] == cluster_index, 'spam_probability'] = spam_probabilities[cluster_name]

        # Menghitung jumlah fitur untuk setiap cluster
        feature_sums = {cluster_name: df[f'{cluster_name}_feature_sum'].sum() for cluster_name in clusters.keys()}

        # Menghitung total dari semua cluster
        total_sum = sum(feature_sums.values())

        # Memanggil fungsi untuk menambahkan fitur dengan probabilitas spam yang baru
        add_cluster_features(df, clusters, spam_probabilities)


        print(df[['Cluster', 'spam_probability', 'cluster_1_feature_sum', 'cluster_2_feature_sum', 'cluster_3_feature_sum']].head())
        
        X_kmeans = pd.concat([df[['spam_probability']], 
                              df[['cluster_1_feature_sum']], 
                              df[['cluster_2_feature_sum']], 
                              df[['cluster_3_feature_sum']], 
                              df[['Cluster']]], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_kmeans)
        imputer_kmeans = SimpleImputer(strategy='most_frequent')
        X_scaled_imputed = imputer_kmeans.fit_transform(X_scaled)

        optimal_clusters = 2
        
        kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=33)
        kmeans_model.fit(X_scaled_imputed)

        df['Cluster'] = kmeans_model.labels_

        return kmeans_model, X_scaled_imputed

    X_train_imputed,X_test_imputed, y_train, y_test, scaler, imputer,X_train, X_test = preprocess_data(df)
    best_model_logisticR = train_logistic_regression(X_train_imputed, y_train)
    kmeans_model, X_scaled_imputed = train_kmeans(df)
    
    def display_dashboard(df, X_train, X_test, y_train, y_test, scaler, imputer, best_model_logisticR,X_scaled_imputed,X_test_imputed, kmeans_model):
        # Use loaded logistic model for predictions
        y_best_predict = best_model_logisticR.predict(X_test_imputed)

        # Title
        st.title("Dashboard Analisis Klasifikasi Email Spam")

        # Sidebar for User Input
        options = [
            "Pilih Opsi untuk Ditampilkan", 
            "Deskripsi Data", 
            "Evaluasi Model", 
            "Visualisasi", 
            "Kesimpulan"
        ]
        selected_option = st.sidebar.selectbox("Pilih Opsi", options)
        st.sidebar.write(f"Selected Creator: {selected_option}")

        # Add table to sidebar
        st.sidebar.subheader("Informasi Perancang")
        user_info = pd.DataFrame({
            "Nama Lengkap": ['Abim Bimasena A.R.P',"Azel Pandya Maheswara N.A", "Danendra Pandya Maheswara",'Sahal Fajri'],
            "JURUSAN": ["S1 Sistem Informasi", "S1 Sistem Informasi", "S1 Sistem Informasi", "S1 Sistem Informasi"],
        })
        st.sidebar.table(user_info)

        # Show content based on selected option
        if selected_option == "Pilih Opsi untuk Ditampilkan":
            st.write("Pilih salah satu opsi untuk melihat informasi lebih lanjut.")

        elif selected_option == "Deskripsi Data":
            st.subheader("Deskripsi Data")
            st.write("""
                Dataset ini berisi informasi tentang frekuensi kata, frekuensi karakter khusus, dan pola penggunaan huruf kapital 
                dalam email. Data ini digunakan untuk mengklasifikasikan email menjadi spam (1) atau bukan spam (0).
                """)

            st.write("**Frekuensi Kata**")
            st.write("""
            1. Kolom: word_freq_make, word_freq_address, word_freq_all, ..., word_freq_conference.
            2. Deskripsi: Kolom ini mencatat frekuensi relatif (dalam persentase) kemunculan kata-kata tertentu di dalam email. Setiap nilai dihitung sebagai 100 kali jumlah kemunculan kata dalam email dibagi dengan total jumlah kata dalam email. Kata-kata ini mencakup berbagai istilah yang sering muncul dalam email spam.
            3. Signifikansi: Kata-kata ini dipilih karena sering muncul dalam pola spam, seperti "free", "business", "credit", atau "money".
            """)

            st.write("**Frekuensi Karakter Khusus**")
            st.write("""
            1. Kolom: char_freq_%3B, char_freq_%28, char_freq_%5B, char_freq_%21, char_freq_%24, char_freq_%23.
            2. Deskripsi: Kolom ini mencatat frekuensi relatif (dalam persentase) kemunculan karakter khusus seperti !, $, atau #. Nilai dihitung sebagai 100 kali jumlah kemunculan karakter khusus dalam email dibagi dengan total karakter dalam email.
            3. Signifikansi: Karakter ini sering digunakan dalam email spam untuk menarik perhatian atau melewati filter spam tradisional.
            """)

            st.write("**Rangkaian Huruf Kapital**")
            st.write("""
            1. Kolom: capital_run_length_average, capital_run_length_longest, capital_run_length_total.
            2. Deskripsi: Kolom ini mencatat pola penggunaan huruf kapital dalam email. capital_run_length_average menunjukkan rata-rata panjang urutan huruf kapital yang tidak terputus, capital_run_length_longest menunjukkan panjang maksimum urutan huruf kapital yang tidak terputus, dan capital_run_length_total mencatat total jumlah huruf kapital dalam email.
            3. Signifikansi: Huruf kapital biasanya digunakan untuk menekankan bagian tertentu dalam email spam.
            """)

            st.write("**Contoh Data**")
            st.dataframe(df.head())
            st.write("**Informasi Umum Dataset**")
            st.write(df.describe())
            st.write("**Jumlah Data yang Hilang**")
            st.write(df.isnull().sum())
            st.write("**Target Kategori Email**")
            
            st.write("""
            Kolom: class.
            Deskripsi: Kolom ini berfungsi sebagai label kategori email, dengan nilai 1 untuk spam dan 0 untuk bukan spam (ham). Ini adalah variabel target yang digunakan dalam model klasifikasi.
            """)

            st.write("**Distribusi Kelas**")
            st.write(df['class'].value_counts())

        elif selected_option == "Evaluasi Model":
            st.subheader("Evaluasi Model")
            st.write(f"**Akurasi Model Terbaik:** {accuracy_score(y_test, y_best_predict):.2f}")
            
            # Input fields for columns from dataset
            with st.form("input_form"):
                st.write("Masukkan nilai untuk fitur-fitur berikut:")
                user_inputs = {column: st.number_input(f"{column}", min_value=0.0, step=0.01) for column in df.columns[:-1]}
                submitted = st.form_submit_button("Prediksi")

            if submitted:
                input_array = np.array([list(user_inputs.values())]).reshape(1, -1)
                input_array_scaled = scaler.transform(input_array)
                input_array_imputed = imputer.transform(input_array_scaled)
                prediction = best_model_logisticR.predict(input_array_imputed)

                st.write("**Hasil Prediksi:**")
                st.write("Email ini termasuk ke dalam **Spam**." if prediction[0] == 1 else "Email ini termasuk ke dalam **Non-Spam**.")

        elif selected_option == "Visualisasi":
            st.subheader("Visualisasi Data dan Model")
            st.write("Visualisasi data dan model yang digunakan untuk analisis klasifikasi email spam.")
            options = [
                "Pilih Opsi untuk Ditampilkan",
                'Logistic Regression Model',
                'K-Means Clustering'
            ]
            selected_viz = st.selectbox("Pilih Model", options)
            
            if selected_viz == 'Logistic Regression Model':
                # Pairplot
                options = [
                    "Pilih Opsi untuk Ditampilkan",
                    'Correlation Heatmap',
                    'ROC Curve',
                    'Precision-Recall Curve',
                    'Confusion Matrix']
                
                st.write(f"**Akurasi Model pada Data Uji:** {accuracy_score(y_test, y_best_predict):.2f}")
                selected_viz = st.selectbox("Pilih Visualisasi", options)
                

                if selected_viz == 'Correlation Heatmap':
                    # Correlation Heatmap
                    st.write("**Correlation Heatmap**")
                    fig = px.imshow(df.corr(), text_auto=True, aspect="auto")
                    st.plotly_chart(fig)

                elif selected_viz == 'ROC Curve':
                    # ROC Curve
                    st.write("**ROC Curve**") 
                    ypd_proba = best_model_logisticR.predict_proba(X_test_imputed)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, ypd_proba)
                    roc_auc = auc(fpr, tpr)
                    st.write(f"**ROC AUC Score:** {roc_auc:.2f}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
                    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    st.plotly_chart(fig)
                
                elif selected_viz == 'Precision-Recall Curve':
                    # Display precision, recall, and f1 score
                    precision_score_value = precision_score(y_test, y_best_predict)
                    recall_score_value = recall_score(y_test, y_best_predict)
                    f1_score_value = f1_score(y_test, y_best_predict)

                    st.write(f"**Precision Score:** {precision_score_value:.2f}")
                    st.write(f"**Recall Score:** {recall_score_value:.2f}")
                    st.write(f"**F1 Score:** {f1_score_value:.2f}")
                    y_prob = best_model_logisticR.predict_proba(X_test)[:, 1]
                    # Precision-Recall Curve
                    st.write("**Precision-Recall Curve**")
                    precision, recall, _ = precision_recall_curve(y_test, y_prob)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve'))
                    fig.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
                    st.plotly_chart(fig)
                
                elif selected_viz == 'Confusion Matrix':
                    # Confusion Matrix
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_best_predict)
                    fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="Actual", color="Count"), x=['Non-Spam', 'Spam'], y=['Non-Spam', 'Spam'], color_continuous_scale='Blues')
                    fig.update_layout(title='Confusion Matrix', coloraxis_showscale=False)
                    st.plotly_chart(fig)
                
            elif selected_viz == 'K-Means Clustering': 
                # st.write(f"**Silhouette Score K-Means Model:** {silhouette_score(X_scaled_imputed, kmeans_model.labels_):.2f}")
                options = [
                    "Pilih Opsi untuk Ditampilkan",
                    'Elbow Method',
                    'Silhouette Score Analysis',
                    'Cluster Plot'
                ]
                selected_viz = st.selectbox("Pilih Visualisasi", options)
                
                if selected_viz == 'Elbow Method':
                    # Elbow Method
                    st.write("**Elbow Method**")
                    inertia = []
                    range_clusters = range(1, 11)
                    for k in range_clusters:
                        kmeans = KMeans(n_clusters=k, random_state=33)
                        kmeans.fit(X_scaled_imputed)
                        inertia.append(kmeans.inertia_)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range_clusters), y=inertia, mode='lines+markers', name='Inertia'))
                    fig.update_layout(title='Elbow Method For Optimal Clusters', xaxis_title='Number of Clusters', yaxis_title='Inertia')
                    st.plotly_chart(fig)

                    st.write("**Inertia Values**")
                    inertia_df = pd.DataFrame({'Number of Clusters': range_clusters, 'Inertia': inertia})
                    st.dataframe(inertia_df)
                    
                    st.write("""Hasil inertia yang diperoleh dari metode elbow menunjukkan nilai inertia (atau jarak total dalam cluster) untuk berbagai jumlah cluster (k) yang digunakan dalam analisis K-Means. Inertia adalah ukuran seberapa baik data dikelompokkan, di mana nilai yang lebih rendah menunjukkan bahwa data dalam cluster lebih dekat satu sama lain. Dalam konteks hasil yang diberikan, dapat melihat bahwa seiring bertambahnya jumlah cluster (dari 1 hingga 10), nilai inertia cenderung menurun.

Pada awalnya, ketika jumlah cluster adalah 1, inertia sangat tinggi, yaitu 12426. Ini menunjukkan bahwa semua data berada dalam satu cluster besar, yang tidak efisien. Ketika jumlah cluster meningkat menjadi 2, inertia turun secara signifikan menjadi 4134.8, menunjukkan bahwa pemisahan data menjadi dua cluster memberikan pengelompokan yang lebih baik. Penurunan inertia terus berlanjut dengan penambahan cluster, tetapi laju penurunan ini mulai melambat setelah mencapai sekitar 4 atau 5 cluster.

Ketika mencapai 5 cluster, nilai inertia adalah 1515.87, dan penurunan lebih lanjut menjadi 1260.18 untuk 6 cluster menunjukkan bahwa meskipun ada penurunan, itu tidak sebanding dengan penurunan yang terjadi pada jumlah cluster yang lebih rendah. Ini adalah indikasi bahwa mungkin telah mencapai titik di mana menambah lebih banyak cluster tidak memberikan peningkatan yang signifikan dalam pengelompokan data.
                             """)

                elif selected_viz == 'Silhouette Score Analysis':
                    # Silhouette Score Analysis

                    silhouette_scores = []
                    for k in range(2, 11):
                        kmeans = KMeans(n_clusters=k, random_state=33)
                        kmeans.fit(X_scaled_imputed)
                        score = silhouette_score(X_scaled_imputed, kmeans.labels_)
                        silhouette_scores.append(score)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(2, 11)), y=silhouette_scores, mode='lines+markers', name='Silhouette Score', line=dict(color='green')))
                    fig.update_layout(title='Silhouette Scores For Optimal Clusters', xaxis_title='Number of Clusters', yaxis_title='Silhouette Score')
                    st.plotly_chart(fig)
                    st.write(f"Silhouette Score: {max(silhouette_scores):.2f}")
                    
                    st.write("**Silhouette Scores**")
                    silhouette_df = pd.DataFrame({'Number of Clusters': range(2, 11), 'Silhouette Score': silhouette_scores})
                    st.dataframe(silhouette_df)
                    
                    st.write("""Hasil skor siluet yang diperoleh menunjukkan variasi dalam kualitas pengelompokan data berdasarkan jumlah kluster yang digunakan. Skor siluet tertinggi, yaitu 0.5667, dicapai dengan dua kluster, 
                             yang mengindikasikan bahwa pengelompokan ini memberikan pemisahan yang baik antara kluster dan jarak yang cukup jauh antar titik data dalam kluster yang berbeda. Namun, seiring bertambahnya jumlah kluster,
                             skor siluet cenderung menurun, dengan skor terendah 0.3220 pada sepuluh kluster. Penurunan ini menunjukkan bahwa pengelompokan yang lebih banyak tidak selalu menghasilkan pemisahan yang lebih baik, 
                             dan bisa jadi menyebabkan kluster yang terlalu kecil atau tumpang tindih, sehingga mengurangi kualitas pengelompokan.""")

                elif selected_viz == 'Cluster Plot':
                    # Ensure 'Cluster' column is added to the DataFrame
                    if 'Cluster' not in df.columns:
                        df['Cluster'] = kmeans_model.labels_

                    pca = PCA(n_components=3)
                    pca_result = pca.fit_transform(X_scaled_imputed)

                    optimal_clusters = 2
                    # Menambahkan hasil PCA ke DataFrame
                    df['PCA1'] = pca_result[:, 0]
                    df['PCA2'] = pca_result[:, 1]
                    
                    fig = go.Figure()

                    for cluster in range(optimal_clusters):
                        fig.add_trace(go.Scatter(
                            x=pca_result[kmeans_model.labels_ == cluster, 0],
                            y=pca_result[kmeans_model.labels_ == cluster, 1],
                            mode='markers',
                            name=f"Cluster {cluster + 1}"
                        ))

                    centroids = np.array([pca_result[kmeans_model.labels_ == i].mean(axis=0) 
                                          for i in range(optimal_clusters)])
                    fig.add_trace(go.Scatter(
                        x=centroids[:, 0],
                        y=centroids[:, 1],
                        mode='markers',
                        marker=dict(size=12, color='Purple', symbol='x'),
                        name='Centroid'
                    ))

                    fig.update_layout(
                        title="SPAM & Non SPAM Cluster Plot",
                        legend_title="Clusters"
                    )

                    st.plotly_chart(fig)
                    st.write("""Dari plot di atas, terlihat bahwa pengelompokan data menggunakan dua kluster memberikan pemisahan yang baik antara data spam dan non-spam. 
                             Kluster 1 (biru) dan kluster 2 (biru cyan) memiliki pemisahan yang jelas, dengan centroid masing-masing menunjukkan pusat kluster.""")
                    st.write("""Dari pembagian cluster menunjukkan bahwa jumlah cluster yang optimal adalah dua, dengan karakteristik yang berbeda di masing-masing cluster. 
                             Cluster 1 didominasi oleh 'cluster_1_feature_sum' dan 'cluster_2_feature_sum' dengan probabilitas spam sebesar 0.7, sedangkan Cluster 2 didominasi oleh 'cluster_3_feature_sum' dengan probabilitas spam yang lebih tinggi, yaitu 0.75. 
                             Dalam hal distribusi data, Cluster 1 memiliki sejumlah data points yang lebih banyak dibandingkan Cluster 2. Visualisasi menggunakan PCA memperlihatkan pemisahan yang jelas antara kedua cluster tersebut. 
                             Selain itu, pusat cluster mencerminkan nilai rata-rata dari fitur-fitur yang ada dalam masing-masing cluster,
                             memberikan gambaran yang lebih jelas mengenai karakteristik data yang terklasifikasi.""")
                    st.write("""Tahap diatas diawali dengan mendefinisikan fitur-fitur yang digunakan untuk setiap cluster dalam sebuah dictionary bernama clusters. Fitur-fitur ini merupakan kolom-kolom dalam dataset yang merepresentasikan karakteristik tertentu, seperti frekuensi kata tertentu (word_freq_*) atau frekuensi karakter khusus (char_freq_*). Setelah itu, dilakukan perhitungan jumlah total nilai fitur pada setiap cluster menggunakan fungsi .sum() untuk mengetahui total kontribusi setiap fitur dalam cluster tersebut. Jumlah total nilai dari semua cluster kemudian dihitung untuk mendapatkan total keseluruhan data fitur yang ada. Berdasarkan total ini, probabilitas spam untuk setiap cluster dihitung dengan membagi jumlah total nilai fitur suatu cluster dengan total keseluruhan nilai fitur.""")
                    st.write(pd.DataFrame(df).head())
                    st.write("""Selanjutnya, kolom-kolom baru ditambahkan ke dalam dataset, seperti cluster_1_feature_sum, cluster_2_feature_sum, dan cluster_3_feature_sum, yang masing-masing merepresentasikan jumlah nilai fitur yang dimiliki oleh setiap cluster untuk setiap baris data. Setelah nilai-nilai ini dihitung, dilakukan proses pengelompokan dengan membandingkan jumlah nilai fitur dari setiap cluster. Cluster dengan jumlah nilai fitur terbesar ditetapkan sebagai cluster untuk baris data tersebut, dan hasilnya disimpan dalam kolom baru bernama Cluster.""")
                    st.write(pd.DataFrame(X_scaled_imputed).head())
                    st.write("""Tahap terakhir adalah memperbarui probabilitas spam untuk setiap cluster sesuai nilai yang telah ditentukan. Fungsi tambahan digunakan untuk menambahkan kolom baru ke dataset yang berisi probabilitas spam berdasarkan cluster yang ditetapkan. Dengan demikian, hasil akhir dari proses ini adalah dataset yang telah diperkaya dengan informasi tentang jumlah fitur dari masing-masing cluster, cluster yang ditetapkan, serta probabilitas spam untuk setiap baris data. Proses ini dirancang untuk mengelompokkan data secara sistematis berdasarkan karakteristik fitur yang telah didefinisikan.""")

        elif selected_option == "Kesimpulan":
            st.subheader("Kesimpulan")
            st.write("""
                        Dapat disimpulkan bahwa ,penelitian ini berhasil mengembangkan model klasifikasi untuk mengelompokkan email menjadi dua kategori, yaitu spam dan non-spam, dengan menggunakan dua metode machine learning: 
                        Logistic Regression dan K-Means Clustering. Model Logistic Regression menunjukkan performa yang sangat baik dengan skor AUC ROC sebesar 0.995, akurasi 99.5%, precision 0.996, recall 0.996, dan F1-score 0.996,
                        yang menandakan kemampuan model dalam mendeteksi email spam dengan tingkat keandalan yang tinggi. Di sisi lain, analisis K-Means Clustering mengindikasikan bahwa pengelompokan dengan dua cluster memberikan pemisahan yang optimal 
                        antara email spam dan non-spam, dengan skor siluet tertinggi 0.5667 pada dua cluster, yang menunjukkan kualitas pengelompokan yang baik. Hasil ini menunjukkan bahwa model clustering tidak hanya efektif dalam mengidentifikasi karakteristik email, 
                        tetapi juga dapat meningkatkan efisiensi proses penyaringan spam. 
                """)

    # Display dashboard
    display_dashboard(df, X_train, X_test, y_train, y_test, scaler, imputer, best_model_logisticR,X_scaled_imputed,X_test_imputed, kmeans_model) 

if __name__ == "__main__":
    main()
