
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, classification_report, make_scorer, confusion_matrix, roc_curve, auc
from itertools import cycle





# Load models and data
@st.cache_resource
def load_models():
    with open('random_forest_stress.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('kmeans_lifestyle.pkl', 'rb') as file:
        kmeans_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return rf_model, kmeans_model, scaler

rf_model, kmeans_model, scaler = load_models()

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/student_lifestyle_dataset.csv')
    X = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day',
            'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day',
            'Extracurricular_Hours_Per_Day', 'GPA']]
    X_scaled = scaler.transform(X)
    df['Cluster'] = kmeans_model.predict(X_scaled)
    return df

def show_home(df):
    st.header("Data Overview")

    # KPI Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Mahasiswa", len(df))
    with col2:
        st.metric("Rata-rata Study Hours", f"{df['Study_Hours_Per_Day'].mean():.1f}")
    with col3:
        st.metric("Rata-rata GPA", f"{df['GPA'].mean():.2f}")

    # Data Distributions
    st.subheader("Distribusi Data")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='GPA', title='Distribusi GPA')
        st.plotly_chart(fig)

    with col2:
        fig = px.bar(df['Stress_Level'].value_counts(),
                    title='Distribusi Tingkat Stress')
        st.plotly_chart(fig)

    # Correlation Matrix using seaborn
    st.subheader("Correlation Matrix")
    corr = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day',
               'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day',
               'GPA']].corr()

    # Create heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

def show_clustering(df):
    # Convert Stress_Level to numeric at the beginning
    stress_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    df['Stress_Level_Numeric'] = df['Stress_Level'].map(stress_mapping)

    st.header("K-Means Clustering Analysis")

    # PCA Visualization
    st.subheader("Visualisasi PCA Cluster")
    X = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day',
            'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day',
            'Extracurricular_Hours_Per_Day', 'GPA']]
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df['Cluster']

    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                    title='PCA Visualization of Clusters',
                    labels={'PC1': 'First Principal Component',
                           'PC2': 'Second Principal Component'},
                    color_continuous_scale='viridis')
    st.plotly_chart(fig)

    # Cluster Characteristics Visualization
    st.subheader("Karakteristik Cluster")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: GPA vs Study Hours
    sns.scatterplot(data=df, x='Study_Hours_Per_Day', y='GPA',
                    hue='Cluster', palette='deep', ax=axes[0,0])
    axes[0,0].set_title('GPA vs Study Hours by Cluster')

    # Plot 2: Sleep Hours vs Stress Level
    sns.scatterplot(data=df, x='Sleep_Hours_Per_Day', y='Stress_Level_Numeric',
                    hue='Cluster', palette='deep', ax=axes[0,1])
    axes[0,1].set_title('Stress Level vs Sleep Hours by Cluster')
    axes[0,1].set_ylabel('Stress Level (0:Low, 1:Moderate, 2:High)')

    # Plot 3: Physical Activity vs Social Hours
    sns.scatterplot(data=df, x='Social_Hours_Per_Day', y='Physical_Activity_Hours_Per_Day',
                    hue='Cluster', palette='deep', ax=axes[1,0])
    axes[1,0].set_title('Physical Activity vs Social Hours by Cluster')

    # Plot 4: Study Hours vs Extracurricular Hours
    sns.scatterplot(data=df, x='Study_Hours_Per_Day', y='Extracurricular_Hours_Per_Day',
                    hue='Cluster', palette='deep', ax=axes[1,1])
    axes[1,1].set_title('Extracurricular vs Study Hours by Cluster')

    plt.tight_layout()
    st.pyplot(fig)

    # Cluster Distribution
    st.subheader("Distribusi Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig = px.bar(cluster_counts,
                 title='Jumlah Mahasiswa per Cluster',
                 labels={'value': 'Jumlah Mahasiswa', 'index': 'Cluster'})
    st.plotly_chart(fig)

    # Cluster Statistics
    st.subheader("Statistik Cluster")
    cluster_stats = df.groupby('Cluster')[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day',
                                         'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day',
                                         'Extracurricular_Hours_Per_Day', 'GPA']].mean()
    st.table(cluster_stats.round(2))

    # Cluster Interpretations
    st.subheader("Interpretasi Cluster")
    cluster_interpretations = {
        0: "Academic Focused: Waktu belajar tinggi, GPA tinggi",
        1: "Balanced Achiever: Aktivitas seimbang, tidur cukup",
        2: "Social Active: Aktivitas sosial tinggi, GPA moderat",
        3: "Struggling Student: Waktu belajar rendah, stress tinggi"
    }

    for cluster, interpretation in cluster_interpretations.items():
        st.write(f"**Cluster {cluster}:** {interpretation}")

def show_feature_importance(df):
    st.header("Feature Importance Analysis")

    # Feature importance bar chart
    feature_importance = pd.DataFrame({
        'feature': ['Study Hours', 'Sleep Hours', 'Social Hours',
                   'Physical Activity', 'Extracurricular', 'GPA'],
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig = px.bar(feature_importance, x='importance', y='feature',
                 title='Feature Importance in Stress Prediction')
    st.plotly_chart(fig)

def predict_stress(study_hours, sleep_hours, social_hours, physical_activity, extracurricular, gpa):
    input_data = np.array([[study_hours, sleep_hours, social_hours,
                           physical_activity, extracurricular, gpa]])
    scaled_input = scaler.transform(input_data)
    prediction = rf_model.predict(scaled_input)
    probabilities = rf_model.predict_proba(scaled_input)
    return prediction[0], probabilities[0]

def predict_cluster(study_hours, sleep_hours, social_hours, physical_activity, extracurricular, gpa):
    input_data = np.array([[study_hours, sleep_hours, social_hours,
                           physical_activity, extracurricular, gpa]])
    scaled_input = scaler.transform(input_data)
    cluster = kmeans_model.predict(scaled_input)
    return cluster[0]


def show_prediction(df):
    show_feature_importance(df)

    st.header("Stress Level Prediction")

    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Study Hours", 0.0, 12.0, 6.0)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
        social_hours = st.slider("Social Hours", 0.0, 12.0, 2.0)

    with col2:
        physical_activity = st.slider("Physical Activity", 0.0, 6.0, 1.0)
        extracurricular = st.slider("Extracurricular", 0.0, 6.0, 1.0)
        gpa = st.slider("GPA", 0.0, 4.0, 3.0)

    if st.button("Predict"):
        stress_pred, stress_prob = predict_stress(study_hours, sleep_hours, social_hours,
                                                physical_activity, extracurricular, gpa)
        cluster_pred = predict_cluster(study_hours, sleep_hours, social_hours,
                                     physical_activity, extracurricular, gpa)

        # Results section
        st.subheader("Hasil Prediksi")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Predicted Stress Level:** {stress_pred}")
            st.write(f"**Predicted Cluster:** {cluster_pred}")

            # Probability interpretation
            highest_prob_idx = np.argmax(stress_prob)
            stress_levels = ['Low', 'Moderate', 'High']
            st.write("\n**Probability Analysis:**")
            for level, prob in zip(stress_levels, stress_prob):
                st.write(f"{level}: {prob:.2%}")

        with col2:
            # Probability distribution
            prob_df = pd.DataFrame({
                'Stress Level': ['Low', 'Moderate', 'High'],
                'Probability': stress_prob
            })
            fig = px.bar(prob_df, x='Stress Level', y='Probability',
                        title='Prediction Probabilities')
            st.plotly_chart(fig)

        # Recommendations based on predicted stress level
        st.subheader("Rekomendasi")

        recommendations = {
            'Low': [
                "Pertahankan pola aktivitas sekarang",
                "Tetap jaga keseimbangan waktu",
                "Evaluasi berkala untuk memastikan konsistensi"
            ],
            'Moderate': [
                "Tingkatkan waktu tidur",
                "Atur ulang jadwal kegiatan",
                "Pertimbangkan mengurangi beban aktivitas",
                "Tambah waktu olahraga untuk relaksasi"
            ],
            'High': [
                "Prioritaskan waktu istirahat yang cukup",
                "Kurangi beban kegiatan non-esensial",
                "Tambah aktivitas fisik untuk mengurangi stress",
                "Pertimbangkan konsultasi dengan konselor akademik",
                "Buat jadwal yang lebih terstruktur"
            ]
        }

        # Show recommendations based on predicted stress level
        st.write(f"**Berdasarkan tingkat stress yang diprediksi ({stress_pred}), berikut rekomendasinya:**")
        for rec in recommendations[stress_pred]:
            st.write(f"✓ {rec}")

        # Additional lifestyle insights
        st.write("\n**Insight Tambahan:**")
        if sleep_hours < 7:
            st.write("⚠️ Jam tidur Anda kurang dari rekomendasi minimal (7 jam)")
        if study_hours > 8:
            st.write("⚠️ Waktu belajar tinggi - pastikan ada waktu istirahat yang cukup")
        if physical_activity < 1:
            st.write("⚠️ Tingkatkan aktivitas fisik untuk manajemen stress yang lebih baik")

def show_recommendations(df):
    st.header("Rekomendasi")

    st.subheader("Tips Umum")
    tips = [
        "Atur jadwal belajar yang terstruktur",
        "Pastikan tidur yang cukup (7-8 jam)",
        "Seimbangkan kegiatan akademik dan sosial",
        "Rutin berolahraga minimal 30 menit per hari",
        "Ikuti kegiatan ekstrakurikuler yang diminati"
    ]

    for tip in tips:
        st.write(f"✓ {tip}")

    st.subheader("Rekomendasi Berdasarkan Cluster")
    cluster_recs = {
        "Academic Focused": [
            "Tingkatkan waktu istirahat",
            "Tambah aktivitas sosial",
            "Jaga keseimbangan"
        ],
        "Balanced": [
            "Pertahankan pola yang sekarang",
            "Evaluasi kegiatan secara berkala",
            "Set target yang realistis"
        ],
        "Struggling": [
            "Buat jadwal belajar terstruktur",
            "Tingkatkan kualitas tidur",
            "Cari bantuan bila diperlukan"
        ]
    }

    for cluster, recs in cluster_recs.items():
        st.write(f"**{cluster}:**")
        for rec in recs:
            st.write(f"• {rec}")

def main():
    st.title("Analisis Gaya Hidup Mahasiswa")

    # Sidebar
    st.sidebar.title("Analisis Gaya Hidup Mahasiswa")
    page = st.sidebar.radio("Menu",
                           ["Home",
                            "Clustering Analysis",

                            "Prediction",
                            "Rekomendasi"])

    # Load data
    df = load_data()

    # Page navigation
    if page == "Home":
        show_home(df)
    elif page == "Clustering Analysis":
        show_clustering(df)
    elif page == "Feature Importance":
        show_feature_importance(df)
    elif page == "Prediction":
        show_prediction(df)
    elif page == "Rekomendasi":
        show_recommendations(df)

if __name__ == "__main__":
    main()
