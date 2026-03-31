import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Design & Header der Web-App
st.set_page_config(page_title="The AI Guardian", layout="centered")
st.title("🛡️ The AI Guardian")
st.markdown("**Proof of Concept: ML-gestützte Anomalieerkennung für regulatorisches Reporting**")
st.write("Diese App simuliert den Output eines T-SQL Feature Stores. Klicken Sie auf den Button, um die KI-Prüfung zu starten.")

# 2. Daten laden (Direkt im Code integriert, damit keine externen Dateien fehlen können)
data = {
    'ContractID': ['C-1001', 'C-1002', 'C-1003', 'C-1004', 'C-1005'],
    'Feature_AmountDeviation_Score': [-0.5, -0.2, 0.0, 3.85, -0.1], 
    'Feature_InterestDuration_Ratio': [0.029, 0.017, 0.085, 0.125, 0.016],
    'Feature_RiskClass_Encoded': [1, 1, 2, 3, 1]
}
df = pd.DataFrame(data)

st.subheader("1. Ausgangslage: DWH Feature Store")
st.dataframe(df, use_container_width=True)

st.divider()

# 3. Der interaktive Button
if st.button("🚀 KI-Validierung starten", type="primary"):
    
    with st.spinner("Isolation Forest Algorithmus analysiert multidimensionale Muster..."):
        
        # Das Machine Learning Modell
        model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
        features = ['Feature_AmountDeviation_Score', 'Feature_InterestDuration_Ratio', 'Feature_RiskClass_Encoded']
        X = df[features]
        
        # Vorhersage generieren
        df['Anomaly_Flag'] = model.fit_predict(X)
        df['Is_Anomaly'] = df['Anomaly_Flag'].apply(lambda x: '🚨 Anomalie' if x == -1 else '✅ Normal')
        df['Anomaly_Score'] = model.decision_function(X)
        
    st.subheader("2. Ergebnis der automatisierten Prüfung")
    
    # Filtern und Anzeigen der Ergebnisse
    anomalies = df[df['Anomaly_Flag'] == -1]
    
    if not anomalies.empty:
        st.error(f"Achtung: {len(anomalies)} kritische Datensätze identifiziert!")
        st.dataframe(anomalies[['ContractID', 'Is_Anomaly', 'Anomaly_Score']], use_container_width=True)
        st.info("💡 Business Value: Diese Datensätze weichen stark vom gelernten Muster ab und sollten vor der EZB/BaFin-Meldung durch das Fachabteilungs-Team geprüft werden.")
    else:
        st.success("Alle Datensätze entsprechen den historischen Normen.")
