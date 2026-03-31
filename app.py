import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# --- Design & Header ---
st.set_page_config(page_title="The AI Guardian", layout="centered")
st.title("🛡️ The AI Guardian")
st.markdown("**Proof of Concept: End-to-End Pipeline für ML-gestütztes Meldewesen**")

# ==========================================
# SCHRITT 1: Die Rohdaten
# ==========================================
st.subheader("1. Rohdaten (Kernbankensystem / DWH)")
st.write("Vor der KI-Analyse liegen die Daten unstrukturiert und mit absoluten Werten in der Datenbank vor. Hier ist ein Auszug der aktuellen Meldeperiode:")

# 10 simulierte Rohdaten mit einem massiven Ausreißer (Vertrag 47110004)
raw_data = {
    'Vertragsnr': ['47110001', '47110002', '47110003', '47110004', '47110005', '47110006', '47110007', '47110008', '47110009', '47110010'],
    'Kundennummer': ['8001020', '8001021', '8001022', '8001023', '8001024', '8001025', '8001026', '8001027', '8001028', '8001029'],
    'Kreditvolumen_EUR': [150000, 250000, 80000, 4500000, 300000, 120000, 450000, 95000, 210000, 60000], 
    'Zinssatz_Prozent': [3.5, 3.2, 5.1, 1.5, 4.0, 3.8, 2.9, 4.8, 3.4, 5.5],                 
    'Laufzeit_Monate': [120, 180, 60, 12, 240, 120, 240, 84, 180, 60],                    
    'Basel_Risiko_Klasse': ['Low', 'Low', 'Medium', 'High', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium'] 
}
df_raw = pd.DataFrame(raw_data)
st.dataframe(df_raw, use_container_width=True)

st.divider()

# ==========================================
# SCHRITT 2: Feature Engineering (Deine Domäne)
# ==========================================
st.subheader("2. Feature Engineering (Data Transformation)")
st.info("⚙️ **Architektur-Hinweis:** Dieser Schritt wird in der Praxis hochperformant direkt auf der Datenbank via T-SQL (CTEs, Window Functions) ausgeführt. Absolute Werte werden in statistische Abweichungen übersetzt.")

# Die 10 umgewandelten Datensätze, die das ML-Modell verstehen kann
feature_data = {
    'Vertragsnr': ['47110001', '47110002', '47110003', '47110004', '47110005', '47110006', '47110007', '47110008', '47110009', '47110010'],
    'Kundennummer': ['8001020', '8001021', '8001022', '8001023', '8001024', '8001025', '8001026', '8001027', '8001028', '8001029'],
    'Volumen_Abweichung_3M': [-0.51, -0.22, 0.05, 3.85, -0.11, -0.60, 0.35, -0.02, -0.35, 0.15], 
    'Zins_Laufzeit_Index': [0.029, 0.017, 0.085, 0.125, 0.016, 0.031, 0.012, 0.057, 0.018, 0.091],
    'Basel_Risiko_Code': [1, 1, 2, 3, 1, 1, 1, 2, 1, 2]
}
df_features = pd.DataFrame(feature_data)
st.dataframe(df_features, use_container_width=True)

st.divider()

# ==========================================
# SCHRITT 3: Machine Learning
# ==========================================
st.subheader("3. KI-Validierung (Unsupervised Learning)")
st.write("Der Isolation Forest Algorithmus sucht nun in den aufbereiteten Features nach mehrdimensionalen Anomalien, die durch starre SQL-Prüfregeln fallen würden.")

if st.button("🚀 Machine Learning Validierung starten", type="primary"):
    
    with st.spinner("Algorithmus berechnet multidimensionale Pfadlängen..."):
        
        # contamination=0.1 bedeutet, wir erwarten bei 10 Verträgen ca. 1 Anomalie
        model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        
        # Training nur mit den berechneten Features
        features_for_ml = ['Volumen_Abweichung_3M', 'Zins_Laufzeit_Index', 'Basel_Risiko_Code']
        X = df_features[features_for_ml]
        
        df_features['Anomaly_Flag'] = model.fit_predict(X)
        df_features['Is_Anomaly'] = df_features['Anomaly_Flag'].apply(lambda x: '🚨 Anomalie' if x == -1 else '✅ Normal')
        df_features['Anomaly_Score'] = model.decision_function(X)
        
    st.subheader("Ergebnis der automatisierten Prüfung")
    
    anomalies = df_features[df_features['Anomaly_Flag'] == -1]
    
    if not anomalies.empty:
        st.error(f"Achtung: {len(anomalies)} kritischer Datensatz vor Meldeabgabe blockiert!")
        st.dataframe(anomalies[['Vertragsnr', 'Kundennummer', 'Is_Anomaly', 'Anomaly_Score']], use_container_width=True)
        st.warning("💡 **Business Context:** Dieser Vertrag hat eine unlogische Kombination aus extremem Volumen, kurzer Laufzeit und hohem Risiko. Die manuelle Prüfung verhindert hier potenziell eine fehlerhafte EZB-Meldung.")
    else:
        st.success("Alle Datensätze entsprechen den historischen Normen.")
