### 🔴 Live Demo (Web-App)
**[▶️ Klicken Sie hier, um den Proof of Concept direkt im Browser zu testen](https://deine-url-hier.streamlit.app)**

---
# 🛡️ The AI Guardian: ML-gestützte Anomalieerkennung für regulatorisches Reporting

### 🔴 Live Demo (Web-App)
**[▶️ Klicken Sie hier, um den Proof of Concept direkt im Browser zu testen](https://HIER-DEINE-STREAMLIT-URL-EINTRAGEN.streamlit.app)**

---

## 1. Executive Summary & The Business Problem
Die manuellen Korrekturzyklen vor Meldeterminen (z.B. AnaCredit, Finrep) binden in Finanzinstituten massive Ressourcen. Starre, regelbasierte SQL-Prüfungen (Data Quality Gates) fangen bekannte Fehler ab, scheitern aber in der Praxis oft an komplexen, mehrdimensionalen Anomalien. 

Dieser Proof of Concept zeigt, wie klassisches Data Engineering (T-SQL) und Machine Learning (Python) kombiniert werden können, um Ausreißer in Meldedaten **vollautomatisiert** zu identifizieren, bevor sie an die Aufsichtsbehörden (EZB/BaFin) gemeldet werden.

## 2. Die Lösung: Unsupervised Machine Learning
Anstatt zehntausende Zeilen vor Meldeabgabe manuell zu prüfen, nutzt dieser Ansatz den **Isolation Forest Algorithmus** (Scikit-Learn). Er erlernt selbstständig die normalen Strukturen und Varianzen des Portfolios und isoliert Verträge, die unlogische Muster aufweisen (z.B. paradoxe Kombinationen aus extremem Kreditvolumen, kurzer Laufzeit und hohem Risikocode) – ohne dass diese Regeln vorher explizit hartcodiert werden müssen.

## 3. Architektur & Workflow
1. **DWH Feature Engineering (T-SQL):** Rohdaten aus dem Kernbankensystem werden direkt auf der Datenbank in maschinenlesbare und statistisch vergleichbare Metriken übersetzt (z.B. Z-Scores, Ratios). Ein Beispielskript liegt in diesem Repository bei.
2. **Machine Learning Pipeline (Python):** Der Algorithmus bewertet die transformierten Datensätze.
3. **Fachbereichs-Review:** Nur die von der KI identifizierten kritischen Anomalien werden dem Fachbereich zur manuellen Prüfung vorgelegt (Management by Exception).

## 4. Return on Investment (ROI)
* **Prozess-Effizienz:** Massive Reduktion der manuellen Prüfaufwände in den Fachabteilungen.
* **Risikominimierung:** Prävention von Datenqualitäts-Rügen oder teuren Nachmeldungen durch die EZB/Bundesbank.
* **Architektur-Vorteil:** Vorhandene relationale DWH-Strukturen werden strategisch als "Feature Store" für fortgeschrittene KI-Initiativen nutzbar gemacht.
