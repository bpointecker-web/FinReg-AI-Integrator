# ===============================================================================
# Script: Unsupervised Machine Learning zur Identifikation von Meldefehlern
# ===============================================================================

import pandas as pd
from sklearn.ensemble import IsolationForest

print("Starte ML-Validierung der Meldedaten...\n")

# 1. Daten laden (Der simulierte Output aus dem T-SQL Feature Store)
# WICHTIG: Das Skript geht davon aus, dass es aus dem Hauptordner aufgerufen wird.
try:
    df = pd.read_csv('data/feature_store_export.csv')
except FileNotFoundError:
    print("Fehler: CSV-Datei nicht gefunden. Bitte stelle sicher, dass der Pfad 'data/feature_store_export.csv' existiert.")
    exit()

# 2. Modellierung: Isolation Forest initialisieren
# Wir schätzen, dass etwa 10% der Daten Ausreißer sein könnten (für diesen kleinen Demo-Datensatz).
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# 3. Features für das Training auswählen (ohne die ContractID!)
features = ['Feature_AmountDeviation_Score', 'Feature_InterestDuration_Ratio', 'Feature_RiskClass_Encoded']
X = df[features]

# 4. Modell trainieren und Anomalien vorhersagen (-1 = Anomalie, 1 = Normal)
df['Anomaly_Flag'] = model.fit_predict(X)
df['Is_Anomaly'] = df['Anomaly_Flag'].apply(lambda x: 'YES (Check Required)' if x == -1 else 'No')
df['Anomaly_Score'] = model.decision_function(X)

# 5. Ergebnisse filtern und ausgeben
anomalies = df[df['Is_Anomaly'] == 'YES (Check Required)']

print("Validierung abgeschlossen. Folgende Datensätze weichen signifikant vom Muster ab und müssen manuell geprüft werden:")
print("-" * 75)
if not anomalies.empty:
    print(anomalies[['ContractID', 'Is_Anomaly', 'Anomaly_Score', 'Feature_AmountDeviation_Score']])
else:
    print("Keine Anomalien gefunden.")
print("-" * 75)