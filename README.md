# The AI Guardian – ML-Driven Anomaly Detection for Regulatory Reporting

## 1. Executive Summary & The Business Problem

Im hochregulierten Finanzsektor (AnaCredit, Finrep, Basel IV) binden manuelle Datenqualitätsprüfungen massiv Zeit und Ressourcen. Traditionelle, regelbasierte BI-Ansätze (starre SQL-Prüfskripte) stoßen zunehmend an ihre Grenzen:

- Sie finden nur die Fehler, die im Vorfeld explizit als Regel definiert wurden.
- Sie übersehen subtile, mehrdimensionale Dateninkonsistenzen in komplexen Portfolios.
- Sie generieren hohe Raten an "False Positives", die von Business Analysten mühsam manuell aussortiert werden müssen.

Die Folge sind extrem stressige Korrekturzyklen kurz vor dem Meldetermin und ein hohes operatives Risiko für Strafen durch die Aufsichtsbehörden (OeNB, EZB).

## 2. The Solution: AI-Augmented Data Quality

Dieses Projekt demonstriert einen hybriden Ansatz zur automatisierten Fehlererkennung: hochperformantes relationales Data Engineering (T-SQL) kombiniert mit Unsupervised Machine Learning (Python).

Der **Maßstab für „normal" kommt ausschließlich aus validierten historischen Meldungen**: Der Kreditbetrag wird gegen den historischen Portfolio-Durchschnitt gemessen, der Zinssatz gegen den für den jeweiligen Kreditbetrag erwarteten Wert (lineare Regression, da Betrag und Zins historisch korrelieren – kleine Kredite haben höhere Zinsen als große). Die Alarmschwelle ist der schlechteste je validierte historische Score. Der Isolation Forest bewertet Historie und aktuelle Periode gemeinsam (*pooled*), damit auch grobe Einzelausreißer – Werte weit außerhalb des historischen Bereichs – sauber isoliert werden. (Ein rein auf der Historie trainiertes Modell kann solche „out-of-range"-Punkte nicht isolieren; siehe Abschnitt 6.)

## 3. Architecture & Tech Stack

Der Erfolg von Machine Learning im Enterprise-Umfeld scheitert selten am Algorithmus, sondern fast immer an der Datenaufbereitung. Daher liegt der Fokus dieser Architektur auf einer robusten Data Foundation:

- **Feature Engineering (T-SQL, [feature_engineering.sql](feature_engineering.sql)):** Rohdaten werden direkt auf Datenbankebene transformiert. Der **Kreditbetrag** wird gegen Mittelwert/Streuung des **gesamten historischen Portfolios** gemessen. Der **Zinssatz** wird gegen den für den jeweiligen Kreditbetrag **erwarteten Zins** verglichen – berechnet per linearer Regression (Methode der kleinsten Quadrate, von Hand aus den SQL-Summenformeln, da T-SQL keine `REGR_SLOPE`/`REGR_INTERCEPT`-Funktionen kennt), weil Betrag und Zins historisch korrelieren. Eine flache Durchschnittsbetrachtung würde sonst einen für seine Größe unauffälligen Kredit fälschlich als Ausreißer werten oder umgekehrt einen echten Ausreißer übersehen. Alles **ausschließlich aus der historischen Vergleichsperiode**, nie aus dem gerade zu prüfenden Batch. Der Feature Store trägt die Roh-Vertragsdaten (Betrag, Zins, Laufzeit) **und** die abgeleiteten z-Score-Features, sodass das Dashboard die echten Werte zeigen kann.
- **Lokale Reproduktion ohne DB-Verbindung ([generate_feature_store.py](generate_feature_store.py)):** Bildet exakt dieselbe Logik und Mock-Daten wie das SQL-Skript ab, damit die komplette Pipeline lokal lauffähig ist, ohne einen SQL Server aufzusetzen.
- **Machine Learning Engine (Python / Scikit-Learn, [anomaly_detection.py](anomaly_detection.py)):** Isolation Forest bewertet die aktuelle Periode gegen den historischen Maßstab und liefert je geflaggtem Fall eine Begründung **im Klartext des betroffenen Bankenattributs** (z.B. „Zinssatz … weicht extrem vom historischen Normalbereich ab"), statt eines abstrakten Feature-Namens.
- **Export für DWH/BI:** Alle Scores (nicht nur die Anomalien) werden nach `data/anomaly_scores_output.csv` geschrieben – die Grundlage für eine Rückschreibung ins Data Warehouse bzw. den Import in Power BI/Tableau.

## 4. Business Value (ROI)

- **Effizienzsteigerung:** Weniger manuelle Prüfaufwände in den Fachabteilungen, da Analysten primär die von der KI geflaggten Fälle prüfen – inklusive einer nachvollziehbaren Begründung statt eines reinen Scores.
- **Risikominimierung:** Proaktives Auffinden von Dateninkonsistenzen aus Vorsystemen, bevor diese an die Regulatoren gemeldet werden.
- **Skalierbarkeit:** Der Benchmark passt sich mit jeder neuen historischen Periode automatisch an veränderte Portfoliostrukturen an.

## 5. Setup & Ausführung

```bash
pip install -r requirements.txt

# 1. Feature Stores erzeugen (lokale Reproduktion von feature_engineering.sql)
python generate_feature_store.py

# 2. Anomalieerkennung als CLI-Report ausführen
python anomaly_detection.py

# 3. Interaktives Dashboard starten (öffnet sich im Browser auf http://localhost:8501)
python -m streamlit run app.py

# 4. Tests ausführen
pip install pytest
python -m pytest tests/
```

`feature_engineering.sql` ist die fachliche Referenzimplementierung für den Produktivbetrieb gegen ein DWH; `generate_feature_store.py` bildet dieselbe Logik lokal nach und erzeugt die CSVs, mit denen `anomaly_detection.py` arbeitet.

### Dashboard ([app.py](app.py))

Das Streamlit-Dashboard ist die visuelle Ebene über der Pipeline und der lauffähige Prototyp der in Abschnitt 3 beschriebenen BI-Integration: KPI-Kacheln (geprüfte Verträge / Anomalien / Schwellenwert), ein Prüf-Report mit nachvollziehbarer Begründung je geflaggtem Fall, ein Score-Chart mit eingezeichneter Anomalie-Schwelle und ein Sensitivitätsregler, mit dem der Analyst die Prüfschärfe (Perzentil der Historie) live justieren kann. Alle Skripte lösen ihre Datenpfade relativ zum Projektordner auf und sind daher aus jedem Arbeitsverzeichnis lauffähig.

## 6. Bekannte Grenzen (bewusst offen für Weiterentwicklung)

- **Pooled Scoring vs. Massen-Systematikfehler:** Historie und aktuelle Periode werden gemeinsam bewertet, damit grobe Einzelausreißer (z.B. der Zins-Defaultwert `999999`) überhaupt isoliert werden – ein rein auf der Historie trainiertes Isolation-Forest-Modell erkennt solche „out-of-range"-Werte nicht, weil sie den Score des Trainingsrands erben. Kehrseite: Ein *systematischer* Massenfehler (hunderte identische Fehlbuchungen) könnte als eigenes Cluster erscheinen und der Isolation entgehen. In der Praxis kombiniert man dieses ML-Scoring daher mit einfachen Plausibilitäts-/Validitätsregeln.
- Die Anomalie-Schwelle wird aus dem schlechtesten historisch beobachteten Score abgeleitet (robuster als ein pauschaler `contamination`-Parameter, aber bei sehr kleiner Historie empfindlich gegenüber einzelnen Ausreißern in den historischen Referenzdaten).
- Explainability beschränkt sich auf "stärkste Einzelfeature-Abweichung" (Z-Score gegen die historische Verteilung); für den Produktivbetrieb wäre eine SHAP-basierte Erklärung pro Feature-Kombination aussagekräftiger.
- Mock-Daten sind synthetisch und klein dimensioniert; die Robustheit von Benchmark und Modell bei echten Portfoliogrößen (zehntausende Verträge) ist damit nicht validiert.
