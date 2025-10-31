import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🧠 Machine Learning Aplicado à Saúde — Câncer de Mama")
st.markdown("""
Este aplicativo demonstra a aplicação de **Aprendizagem Supervisionada** e **Não Supervisionada**  
em um dataset real de diagnóstico de **Câncer de Mama** (sklearn.datasets).
""")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

st.subheader("📊 Visão Geral dos Dados")
st.write(df.head())

st.subheader("🔍 Estatísticas Descritivas")
st.write(df.describe())

st.subheader("💡 Distribuição das Classes")
st.bar_chart(df['target'].value_counts())

st.header("🎯 Modelo Supervisionado — Random Forest")
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
st.metric("Acurácia do Modelo", f"{acc:.2%}")
st.text(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.header("🌀 Modelo Não Supervisionado — KMeans + PCA")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_pca)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = labels
df_pca['Diagnóstico Real'] = y
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax2)
st.pyplot(fig2)
st.markdown("✅ **Interpretação:** Os grupos formados pelo KMeans se aproximam das classes reais do diagnóstico.")
st.markdown("---")
st.markdown("Desenvolvido por *Pedro Lucas Marques* — Projeto Final: Machine Learning Aplicado à Saúde 🩺")
