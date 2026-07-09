# ===============================================================================
# Script: Unsupervised Machine Learning zur Identifikation von Meldefehlern
# ===============================================================================
# Der Maßstab für "normal" kommt aus validierten historischen Meldedaten
# (data/historical_feature_store.csv): Der Kreditbetrag wird gegen den
# historischen Portfolio-Durchschnitt gemessen, der Zinssatz gegen den für den
# jeweiligen Kreditbetrag erwarteten Wert (Regression, da Betrag und Zins
# historisch korrelieren). Die Alarmschwelle ist der schlechteste je validierte
# historische Score. Das Isolation-Forest-Modell bewertet Historie und
# aktuelle Periode GEMEINSAM (pooled), damit auch grobe Einzelausreißer – Werte weit
# außerhalb des historischen Bereichs – sauber isoliert werden; ein rein auf der
# Historie trainiertes Modell kann solche "out-of-range"-Punkte nicht isolieren, weil
# sie den Score des Trainingsrands erben. (Grenze bei Massen-Systematikfehlern: siehe
# README, Abschnitt "Bekannte Grenzen".)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from generate_feature_store import build_benchmark, expected_interest

FEATURE_COLUMNS = [
    "Feature_AmountDeviation_Score",
    "Feature_InterestDeviation_Score",
]
# Roh-Vertragsdaten, die die Feature-Stores mitliefern – für Anzeige und für die
# menschenlesbare Begründung (echte Werte statt abstrakter Scores).
RAW_DISPLAY_COLUMNS = ["LoanAmount_EUR", "InterestRate", "Duration_Months"]
REQUIRED_COLUMNS = ["ContractID", *RAW_DISPLAY_COLUMNS, *FEATURE_COLUMNS]

# Zuordnung Modell-Feature -> Rohattribut, um eine Abweichung im Klartext des
# jeweiligen Bankenattributs zu erklären.
_FEATURE_ATTRIBUTE = {
    "Feature_AmountDeviation_Score": "amount",
    "Feature_InterestDeviation_Score": "interest",
}

# Pfade relativ zum Skript-Ort auflösen, damit die Pipeline von jedem
# Arbeitsverzeichnis aus lauffähig ist (z.B. aus der Streamlit-Vorschau heraus).
DATA_DIR = Path(__file__).resolve().parent / "data"
HISTORICAL_FEATURE_STORE_PATH = str(DATA_DIR / "historical_feature_store.csv")
CURRENT_FEATURE_STORE_PATH = str(DATA_DIR / "feature_store_export.csv")
OUTPUT_PATH = str(DATA_DIR / "anomaly_scores_output.csv")


def _de_number(value: float, decimals: int) -> str:
    """Formatiert eine Zahl im deutschen Stil (1.234.567,89) – plattformunabhängig
    ohne locale-Abhängigkeit."""
    us = f"{value:,.{decimals}f}"  # z.B. 1,234,567.89
    return us.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def format_eur(value: float) -> str:
    return f"{_de_number(value, 0)} €"


def format_interest(value: float) -> str:
    """Zinssatz als Prozent (0.035 -> '3,50 %')."""
    return f"{_de_number(value * 100, 2)} %"


def load_feature_store(path: str) -> pd.DataFrame:
    # ContractID zwingend als String lesen, sonst gehen führende Nullen der
    # nullgepolsterten Vertragsnummern verloren (00000151723 -> 151723).
    df = pd.read_csv(path, dtype={"ContractID": str})
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"'{path}' fehlen die Spalten: {sorted(missing_columns)}")
    return df


def _severity_word(z_score: float) -> str:
    """Sprachliche Abstufung nach Abweichungsgröße, damit ein grenzwertiger Fall
    (knapp über der Schwelle) nicht dieselbe dramatische Sprache bekommt wie ein
    Wert, der um Größenordnungen daneben liegt."""
    magnitude = abs(z_score)
    if magnitude >= 5:
        return "extrem"
    if magnitude >= 2:
        return "deutlich"
    return "auffällig"


def explain_anomaly(row: pd.Series, train_stats: pd.DataFrame, benchmark: dict) -> str:
    """Erklärt die Abweichung im Klartext des betroffenen Bankenattributs – mit
    echtem Wert und historischem Normalbereich, statt eines abstrakten Feature-Namens."""
    z_scores = (row[FEATURE_COLUMNS] - train_stats["mean"]) / train_stats["std"]
    top_feature = z_scores.abs().idxmax()
    kind = _FEATURE_ATTRIBUTE[top_feature]
    severity = _severity_word(z_scores[top_feature])

    if kind == "interest":
        value = format_interest(row["InterestRate"])
        # Der erwartete Zins hängt vom gemeldeten Kreditbetrag ab (kleine Kredite
        # haben historisch höhere Zinsen als große) – deshalb wird hier explizit
        # der für DIESEN Betrag erwartete Wert genannt, nicht ein pauschaler
        # Durchschnitt über alle Kredithöhen.
        amount = format_eur(row["LoanAmount_EUR"])
        expected = format_interest(expected_interest(row["LoanAmount_EUR"], benchmark))
        reason = (
            f"Zinssatz {value} weicht {severity} vom für diesen Kreditbetrag ({amount}) "
            f"historisch erwarteten Zinssatz ab (erwartet: rund {expected})."
        )
        if row["InterestRate"] > 1.0:  # > 100 % p.a. ist unmöglich
            reason += " Vermutlich ein Platzhalter-/Defaultwert aus dem Vorsystem."
        return reason
    value = format_eur(row["LoanAmount_EUR"])
    avg = format_eur(benchmark["Resolved_Avg_Amount"])
    return f"Kreditbetrag {value} weicht {severity} vom historischen Normalbereich ab (Ø {avg})."


def score_current_period(
    historical_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold_percentile: float = 0.0,
) -> pd.DataFrame:
    """Bewertet die aktuelle Periode gegen den aus der Historie abgeleiteten Maßstab.

    threshold_percentile steuert die Sensitivität: 0.0 (Default) nimmt den
    schlechtesten historisch beobachteten Score als Schwelle (nur klarere
    Ausreißer als alles bisher Validierte werden geflaggt). Höhere Perzentile
    heben die Schwelle an und flaggen bereits Fälle, die schlechter sind als die
    unteren x % der Historie – der Analyst kann so mehr oder weniger streng prüfen.
    """
    # Historie + aktuelle Periode gemeinsam ins Modell (pooled), damit grobe
    # Einzelausreißer isoliert werden können. Der Bezugsrahmen bleibt die Historie:
    # die Schwelle wird ausschließlich aus den historischen Scores abgeleitet.
    pooled = pd.concat([historical_df, current_df], ignore_index=True)
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(pooled[FEATURE_COLUMNS])

    # Schwellenwert aus den historischen Scores ableiten statt einer festen
    # Kontaminationsquote: das vermeidet die Zirkularität von "wir sagen dem Modell,
    # wie viel Prozent Anomalien sind" und bleibt erklärbar. Perzentil 0 ==
    # schlechtester validierter Fall (Default), höhere Perzentile = strengere Prüfung.
    historical_scores = model.decision_function(historical_df[FEATURE_COLUMNS])
    anomaly_threshold = float(np.percentile(historical_scores, threshold_percentile))

    result = current_df.copy()
    result["Anomaly_Score"] = model.decision_function(current_df[FEATURE_COLUMNS])
    result["Is_Anomaly"] = result["Anomaly_Score"] < anomaly_threshold

    train_stats = pd.DataFrame(
        {"mean": historical_df[FEATURE_COLUMNS].mean(), "std": historical_df[FEATURE_COLUMNS].std(ddof=1)}
    )
    # Klassen-Normalbereich (Betrag/Zins) aus derselben Quelle wie das Feature-
    # Engineering, für die Klartext-Begründung.
    benchmark = build_benchmark(historical_df)
    result["Reason"] = result.apply(
        lambda row: explain_anomaly(row, train_stats, benchmark) if row["Is_Anomaly"] else "", axis=1
    )

    result = result.sort_values("Anomaly_Score", ascending=True).reset_index(drop=True)
    # Schwellenwert für Downstream-Konsumenten (z.B. UI-Chart) verfügbar machen,
    # ohne die Rückgabe-Signatur zu ändern.
    result.attrs["anomaly_threshold"] = anomaly_threshold
    return result


def main() -> None:
    print("Starte ML-Validierung der Meldedaten...\n")

    historical_df = load_feature_store(HISTORICAL_FEATURE_STORE_PATH)
    current_df = load_feature_store(CURRENT_FEATURE_STORE_PATH)

    result = score_current_period(historical_df, current_df)
    result.to_csv(OUTPUT_PATH, index=False)

    anomalies = result[result["Is_Anomaly"]]

    print("Validierung abgeschlossen. Folgende Datensätze weichen signifikant vom historischen")
    print("Muster ab und müssen manuell geprüft werden (schwerste Abweichung zuerst):")
    print("-" * 88)
    if not anomalies.empty:
        print(anomalies[["ContractID", "Anomaly_Score", "Reason"]].to_string(index=False))
    else:
        print("Keine Anomalien gefunden.")
    print("-" * 88)
    print(f"\nAlle Scores wurden nach '{OUTPUT_PATH}' geschrieben (Grundlage für DWH-Rückschreibung / BI).")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Fehler: Datei nicht gefunden ({e}). Bitte zuerst 'python generate_feature_store.py' ausführen.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        sys.exit(1)
