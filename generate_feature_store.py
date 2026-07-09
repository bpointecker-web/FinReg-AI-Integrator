# ===============================================================================
# Script: Lokale, DB-freie Reproduktion von feature_engineering.sql
# ===============================================================================
# Bildet exakt dieselbe Logik und Mock-Daten wie feature_engineering.sql ab,
# damit die komplette Pipeline (Feature Engineering -> ML-Scoring) ohne
# SQL-Server-Verbindung lokal nachvollziehbar und reproduzierbar ist.
#
# Erzeugt zwei Feature-Stores:
#   - data/historical_feature_store.csv  -> Trainingsdaten (validierte Historie)
#   - data/feature_store_export.csv      -> Zu prüfende aktuelle Periode
#
# Beide CSVs tragen die Roh-Vertragsdaten (Betrag, Zins, Laufzeit) UND die daraus
# abgeleiteten ML-Features. So kann das Dashboard die echten Werte zeigen, während
# das Modell auf den Features rechnet.
#
# Der Benchmark wird ausschließlich aus der Historie berechnet und auf beide
# Datensätze angewendet - genau wie im SQL-Skript, damit ein Ausreißer in der
# aktuellen Periode seinen eigenen Vergleichsmaßstab nicht verzerren kann.
#
# WICHTIG: Kreditbetrag und Zinssatz hängen historisch stark zusammen (kleine
# Kredite haben höhere Zinsen als große). Der Zins-Normalbereich wird daher NICHT
# als flacher Portfolio-Durchschnitt berechnet (das würde einen für seine Größe
# unauffälligen Kredit fälschlich als Ausreißer werten, oder umgekehrt einen
# echten Ausreißer übersehen), sondern per linearer Regression "Zins in
# Abhängigkeit vom Kreditbetrag" – die Abweichung ist das Residuum dieser
# Regression, nicht die Abweichung vom globalen Mittelwert.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ausgabeverzeichnis relativ zum Skript-Ort, damit der Generator von jedem
# Arbeitsverzeichnis aus dieselben Feature-Stores erzeugt.
DATA_DIR = Path(__file__).resolve().parent / "data"

# Vertragsnummern sind 11-stellige, nullgepolsterte Strings (wie echte
# Kontonummern). WICHTIG: als String halten, sonst gehen die führenden Nullen
# verloren – siehe dtype-Handling in anomaly_detection.load_feature_store.
HISTORICAL_DATA = [
    # ContractID, LoanAmount_EUR, InterestRate, Duration_Months
    ("00000100841", 100000.00, 0.0360, 110),
    ("00000100842", 130000.00, 0.0340, 130),
    ("00000100843", 150000.00, 0.0355, 100),
    ("00000100844", 180000.00, 0.0330, 150),
    ("00000100845", 200000.00, 0.0345, 160),
    ("00000100846", 220000.00, 0.0325, 200),
    ("00000100847", 260000.00, 0.0310, 220),
    ("00000100848", 300000.00, 0.0300, 240),
    ("00000100849", 60000.00, 0.0500, 55),
    ("00000100850", 75000.00, 0.0490, 65),
    ("00000100851", 90000.00, 0.0480, 70),
    ("00000100852", 110000.00, 0.0470, 80),
    ("00000100853", 140000.00, 0.0460, 90),
    ("00000100854", 170000.00, 0.0450, 95),
    ("00000100855", 700000.00, 0.0220, 24),
    ("00000100856", 850000.00, 0.0200, 30),
    ("00000100857", 950000.00, 0.0210, 36),
    ("00000100858", 1100000.00, 0.0190, 40),
    ("00000100859", 1400000.00, 0.0180, 48),
]

CURRENT_PERIOD_DATA = [
    # ContractID, LoanAmount_EUR, InterestRate, Duration_Months
    ("00000151721", 150000.00, 0.0350, 120),
    ("00000151722", 250000.00, 0.0320, 180),
    ("00000151723", 200000.00, 999999.00, 120),  # ANOMALIE: Zins-Defaultwert 999999 (Platzhalter aus Vorsystem)
    ("00000151724", 80000.00, 0.0510, 60),
    ("00000151725", 300000.00, 0.0400, 240),
    ("00000151726", 140000.00, 0.0360, 100),
    ("00000151727", 200000.00, 0.0330, 150),
    ("00000151728", 95000.00, 0.0480, 70),
    ("00000151729", 105000.00, 0.0460, 80),
    ("00000151730", 170000.00, 0.0370, 130),
]

RAW_COLUMNS = ["ContractID", "LoanAmount_EUR", "InterestRate", "Duration_Months"]


def build_benchmark(historical_df: pd.DataFrame) -> dict:
    """Berechnet den historischen Normalbereich für Kreditbetrag und Zinssatz.

    Kreditbetrag: Mittelwert/StdAbw. über das gesamte historische Portfolio.
    Zinssatz: linear abhängig vom Kreditbetrag modelliert (Regression), weil
    Zins und Betrag historisch korrelieren – der erwartete Zins ist also
    "Zins für einen Kredit dieser Höhe", nicht der globale Durchschnitt.

    Einzige Quelle für den Normalbereich – genutzt vom Feature-Engineering und
    von der menschenlesbaren Begründung im Scoring.
    """
    slope, intercept = np.polyfit(historical_df["LoanAmount_EUR"], historical_df["InterestRate"], 1)
    predicted_interest = intercept + slope * historical_df["LoanAmount_EUR"]
    residuals = historical_df["InterestRate"] - predicted_interest
    # ddof=2, da zwei Regressionsparameter (Steigung, Achsenabschnitt) geschätzt wurden.
    residual_std = np.sqrt((residuals**2).sum() / (len(residuals) - 2))

    return {
        "Resolved_Avg_Amount": historical_df["LoanAmount_EUR"].mean(),
        "Resolved_StDev_Amount": historical_df["LoanAmount_EUR"].std(ddof=1),
        "Interest_Regression_Slope": slope,
        "Interest_Regression_Intercept": intercept,
        "Resolved_StDev_InterestResidual": residual_std,
    }


def expected_interest(amount: float, benchmark: dict) -> float:
    """Für einen gegebenen Kreditbetrag historisch erwarteter Zinssatz (Regressionsgerade)."""
    return benchmark["Interest_Regression_Intercept"] + benchmark["Interest_Regression_Slope"] * amount


def _z_score(values: pd.Series, avg: float, std: float) -> pd.Series:
    """z-Score mit sicherem Umgang mit Streuung 0 (dann 0.0)."""
    if not std or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - avg) / std


def compute_features(raw_df: pd.DataFrame, benchmark: dict) -> pd.DataFrame:
    amount_deviation = _z_score(
        raw_df["LoanAmount_EUR"], benchmark["Resolved_Avg_Amount"], benchmark["Resolved_StDev_Amount"]
    )
    # Zins-Abweichung als Regressions-Residuum (Ist-Zins minus für DIESEN Betrag
    # erwarteter Zins), nicht als Abweichung vom globalen Durchschnitt.
    interest_residual = raw_df["InterestRate"] - expected_interest(raw_df["LoanAmount_EUR"], benchmark)
    interest_deviation = _z_score(interest_residual, 0.0, benchmark["Resolved_StDev_InterestResidual"])

    # Roh-Vertragsdaten + abgeleitete Features nebeneinander ausgeben.
    return pd.DataFrame(
        {
            "ContractID": raw_df["ContractID"],
            "LoanAmount_EUR": raw_df["LoanAmount_EUR"],
            "InterestRate": raw_df["InterestRate"],
            "Duration_Months": raw_df["Duration_Months"],
            "Feature_AmountDeviation_Score": amount_deviation.round(4),
            "Feature_InterestDeviation_Score": interest_deviation.round(4),
        }
    )


def main() -> None:
    historical_df = pd.DataFrame(HISTORICAL_DATA, columns=RAW_COLUMNS)
    current_df = pd.DataFrame(CURRENT_PERIOD_DATA, columns=RAW_COLUMNS)

    benchmark = build_benchmark(historical_df)

    historical_features = compute_features(historical_df, benchmark)
    current_features = compute_features(current_df, benchmark)

    DATA_DIR.mkdir(exist_ok=True)
    historical_features.to_csv(DATA_DIR / "historical_feature_store.csv", index=False)
    current_features.to_csv(DATA_DIR / "feature_store_export.csv", index=False)

    print("Feature Stores erzeugt:")
    print("  data/historical_feature_store.csv (Training,", len(historical_features), "Datensätze)")
    print("  data/feature_store_export.csv     (Scoring,", len(current_features), "Datensätze)")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Fehler: Verzeichnis 'data/' nicht gefunden ({e}).", file=sys.stderr)
        sys.exit(1)
