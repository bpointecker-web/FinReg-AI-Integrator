# ===============================================================================
# Script: Streamlit-Dashboard für die ML-Anomalieerkennung im Meldewesen
# ===============================================================================
# Interaktive Vorschau der Pipeline: lädt die Feature-Stores, bewertet die
# aktuelle Meldeperiode gegen das auf der Historie trainierte Modell und
# präsentiert die Ergebnisse so, wie es eine Fachabteilung in einem BI-Dashboard
# erwarten würde – inklusive Sensitivitätsregler und nachvollziehbarer Begründung
# je geflaggtem Fall.
#
# Start:  streamlit run app.py

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from anomaly_detection import (
    CURRENT_FEATURE_STORE_PATH,
    HISTORICAL_FEATURE_STORE_PATH,
    format_eur,
    format_interest,
    load_feature_store,
    score_current_period,
)
from generate_feature_store import build_benchmark, expected_interest
from generate_feature_store import main as generate_feature_stores

st.set_page_config(page_title="AI Guardian – Meldewesen-Validierung", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt beide Feature-Stores. Fehlen sie – etwa bei einem frischen Deploy, wo nur
    der Repo-Inhalt (ohne generierte CSVs) vorliegt –, werden sie zuvor aus den
    Rohdaten erzeugt. Die App ist damit ohne manuellen Vorlauf lauffähig.
    Gecached: läuft nur einmal je Session."""
    if not (os.path.exists(HISTORICAL_FEATURE_STORE_PATH) and os.path.exists(CURRENT_FEATURE_STORE_PATH)):
        generate_feature_stores()
    historical_df = load_feature_store(HISTORICAL_FEATURE_STORE_PATH)
    current_df = load_feature_store(CURRENT_FEATURE_STORE_PATH)
    return historical_df, current_df


def highlight_flagged_rows(row: pd.Series) -> list[str]:
    """Färbt Zeilen mit Prüf-Markierung (🚩 in der Status-Spalte) dezent rot ein."""
    color = "background-color: rgba(214, 39, 40, 0.18)" if "🚩" in str(row["Status"]) else ""
    return [color] * len(row)


# --- Header ---------------------------------------------------------------------
st.title("🛡️ AI Guardian – ML-Validierung der Meldedaten")
st.caption(
    "Unsupervised Anomalieerkennung (Isolation Forest) für regulatorische Meldungen. "
    "Das Modell lernt aus validierten Vorperioden und flaggt Verträge, die signifikant "
    "vom historischen Muster abweichen – lange vor dem Meldetermin."
)

# --- Intro / Onboarding ---------------------------------------------------------
# Holt einen projektfremden Betrachter ab: Was ist das, wofür, wie liest man es?
st.info(
    "**Worum geht es hier?** Banken müssen regelmäßig große Datenmengen an die Aufsicht "
    "melden (AnaCredit, FinRep, Basel). Fehlerhafte Meldungen verursachen Korrekturstress "
    "kurz vor der Frist und riskieren Strafen. Dieses Dashboard prüft eine komplette "
    "Meldeperiode automatisch mit Machine Learning: Statt starrer Wenn-Dann-Regeln lernt das "
    "Modell aus geprüften Vergangenheitsdaten, wie *normale* Verträge aussehen, und markiert "
    "auffällige Ausreißer zur gezielten manuellen Kontrolle – **bevor** gemeldet wird.  \n"
    "_Hinweis: Alle hier gezeigten Daten sind synthetische Demo-Daten._"
)

st.markdown("**So funktioniert's – in drei Schritten:**")
step1, step2, step3 = st.columns(3)
with step1:
    with st.container(border=True):
        st.markdown("**① Lernen**")
        st.caption(
            "Das Modell lernt aus validierten Meldungen abgeschlossener Vorperioden, "
            "wie unauffällige Verträge aussehen (das „Normalbild“)."
        )
with step2:
    with st.container(border=True):
        st.markdown("**② Prüfen**")
        st.caption(
            "Jeder Vertrag der aktuellen Periode wird gegen dieses Normalbild bewertet "
            "und erhält einen Auffälligkeits-Wert (Anomaly Score)."
        )
with step3:
    with st.container(border=True):
        st.markdown("**③ Erklären**")
        st.caption(
            "Auffällige Verträge werden markiert – jeweils mit Begründung, welches "
            "Merkmal am stärksten vom historischen Muster abweicht."
        )

with st.expander("ℹ️ Hintergrund & Business-Kontext (warum das relevant ist)"):
    st.markdown(
        "**Das Problem mit klassischen Prüfregeln:** Starre SQL-Prüfskripte finden nur Fehler, "
        "die vorab als Regel definiert wurden, übersehen subtile mehrdimensionale "
        "Inkonsistenzen und erzeugen viele *False Positives*, die Analysten mühsam aussortieren.\n\n"
        "**Der Mehrwert dieses Ansatzes:**\n"
        "- **Effizienz:** Analysten prüfen nur noch die wenigen von der KI markierten Fälle – "
        "inklusive nachvollziehbarer Begründung statt eines nackten Scores.\n"
        "- **Risiko:** Fehlerhafte Datenanlieferungen aus Vorsystemen werden proaktiv gefunden, "
        "bevor sie an OeNB/EZB gemeldet werden.\n"
        "- **Skalierbarkeit:** Der Vergleichsmaßstab passt sich mit jeder neuen Vorperiode an "
        "veränderte Portfolios an – ohne hunderte Regeln manuell umzuschreiben.\n\n"
        "**Technisch dahinter:** T-SQL-Feature-Engineering (Portfolio-Benchmark für Betrag/Zins) → "
        "Isolation-Forest-Training auf der Historie → Scoring der aktuellen Periode → Export "
        "der Scores zurück ins DWH/BI. Dieses Dashboard ist die visuelle Spitze dieser Pipeline."
    )

# --- Daten laden ----------------------------------------------------------------
try:
    historical_df, current_df = load_data()
except FileNotFoundError:
    st.error(
        "Feature-Stores nicht gefunden. Bitte zuerst im Projektordner "
        "`python generate_feature_store.py` ausführen."
    )
    st.stop()

# --- Sidebar: Steuerung ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Prüf-Einstellungen")
    threshold_percentile = st.slider(
        "Sensitivität (Perzentil der Historie)",
        min_value=0,
        max_value=30,
        value=0,
        step=5,
        help=(
            "0 = nur Fälle, die klarer abweichen als der schlechteste jemals validierte "
            "Vertrag (streng, wenige Fehlalarme). Höhere Werte prüfen zunehmend strenger "
            "und flaggen auch grenzwertige Fälle."
        ),
    )
    st.caption("Regler höher schieben = strengere Prüfung → mehr Verträge werden markiert.")
    st.divider()
    st.metric("Historie (Referenz)", f"{len(historical_df)} Verträge")
    st.metric("Aktuelle Periode", f"{len(current_df)} Verträge")
    st.caption(
        "Der Maßstab für „normal“ kommt ausschließlich aus der validierten Historie – "
        "die Alarmschwelle ist der schlechteste je geprüfte historische Vertrag."
    )

# --- Scoring --------------------------------------------------------------------
result = score_current_period(historical_df, current_df, threshold_percentile=float(threshold_percentile))
threshold = result.attrs["anomaly_threshold"]
anomalies = result[result["Is_Anomaly"]]
# Portfolio-Normalbereich (Betrag/Zins) für den Fokus-Vergleich im Prüf-Report.
benchmark = build_benchmark(historical_df)

# --- KPI-Kacheln ----------------------------------------------------------------
st.subheader("Ergebnis dieser Meldeperiode")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Geprüfte Verträge",
    len(result),
    help="Anzahl Verträge in der aktuellen Meldeperiode, die das Modell bewertet hat.",
)
col2.metric(
    "🚩 Anomalien",
    len(anomalies),
    help="Verträge, die stärker vom historischen Muster abweichen als die gelernte Schwelle "
    "– sie gehen in die manuelle Prüfung.",
)
col3.metric(
    "Quote",
    f"{len(anomalies) / len(result):.0%}",
    help="Anteil der markierten Verträge an allen geprüften. Im Meldewesen typischerweise "
    "niedrig (oft nur 1–2 %).",
)
col4.metric(
    "Schwellenwert",
    f"{threshold:.4f}",
    help="Aus der Historie gelernte Grenze: Verträge mit einem Anomaly Score darunter werden "
    "markiert. Der Wert wird nicht willkürlich gesetzt, sondern vom schlechtesten je "
    "validierten Fall abgeleitet.",
)
st.caption(
    "So liest du die Kacheln: Von allen geprüften Verträgen fällt der markierte Anteil "
    "(„Anomalien“) unter die aus der Vergangenheit gelernte Schwelle – nur diese müssen "
    "Menschen noch anschauen."
)

st.divider()

# --- Anomalie-Report ------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("📋 Prüf-Report")
    if anomalies.empty:
        st.success(
            "Keine Anomalien – alle Verträge liegen im historischen Normbereich. "
            "Nichts zu prüfen. ✅"
        )
    else:
        top = anomalies.iloc[0]
        st.warning(
            f"**{len(anomalies)} von {len(result)} Verträgen** zur manuellen Prüfung markiert "
            f"(schwerste zuerst). Auffälligster Fall: **{top['ContractID']}** – {top['Reason']}."
        )
        st.caption(
            "Was heißt „markiert“? Diese Verträge weichen ungewöhnlich stark ab und sollten "
            "gegen das liefernde Vorsystem geprüft werden – dahinter steckt oft ein Zahlendreher "
            "oder ein Mapping-Fehler, nicht zwingend Betrug."
        )
        for _, row in anomalies.iterrows():
            with st.container(border=True):
                st.markdown(f"**Vertrag {row['ContractID']}**  ·  Score `{row['Anomaly_Score']:.4f}`")
                st.markdown(f"↳ {row['Reason']}")
                # Fokus-Vergleich: gemeldeter Wert vs. historischer Normalwert
                mc1, mc2 = st.columns(2)
                if "Zinssatz" in row["Reason"]:
                    mc1.metric("Zinssatz (gemeldet)", format_interest(row["InterestRate"]))
                    mc2.metric(
                        f"Erwartet für {format_eur(row['LoanAmount_EUR'])}",
                        format_interest(expected_interest(row["LoanAmount_EUR"], benchmark)),
                    )
                elif "Kreditbetrag" in row["Reason"]:
                    mc1.metric("Kreditbetrag (gemeldet)", format_eur(row["LoanAmount_EUR"]))
                    mc2.metric("Historisch üblich", format_eur(benchmark["Resolved_Avg_Amount"]))

with right:
    st.subheader("📊 Score-Verteilung")
    chart_df = result.assign(Status=result["Is_Anomaly"].map({True: "Anomalie", False: "Normal"}))
    bars = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Anomaly_Score:Q", title="Anomaly Score (niedriger = auffälliger)"),
            y=alt.Y("ContractID:N", sort="x", title=None),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Normal", "Anomalie"], range=["#4c78a8", "#d62728"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["ContractID", "Anomaly_Score", "Reason"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="#d62728", strokeDash=[4, 4])
        .encode(x="t:Q")
    )
    st.altair_chart(bars + rule, use_container_width=True)
    st.caption("Rote gestrichelte Linie = Anomalie-Schwellenwert. Balken links davon werden geflaggt.")

st.divider()

# --- Meldedaten-Tabelle (echte Vertragswerte) + Export --------------------------
st.subheader("🗂️ Aktuelle Meldedaten (geprüft)")
st.caption(
    "Die tatsächlich gemeldeten Vertragsdaten dieser Periode. Markierte Zeilen weichen vom "
    "historischen Muster ab – Details und Begründung dazu im Prüf-Report oben."
)
meldedaten = pd.DataFrame(
    {
        "Vertragsnummer": result["ContractID"],
        "Kreditbetrag": result["LoanAmount_EUR"].map(format_eur),
        "Zinssatz": result["InterestRate"].map(format_interest),
        "Laufzeit (Monate)": result["Duration_Months"],
        "Anomaly-Score": result["Anomaly_Score"].map(lambda v: f"{v:.4f}"),
        "Status": result["Is_Anomaly"].map(lambda flag: "🚩 Prüfen" if flag else "✓ ok"),
    }
)
st.dataframe(
    meldedaten.style.apply(highlight_flagged_rows, axis=1),
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "⬇️ Ergebnis als CSV herunterladen (inkl. Feature-Werten)",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="anomaly_scores_output.csv",
    mime="text/csv",
)

with st.expander("🔧 Technische Feature-Werte (Eingabe des Modells)"):
    st.caption(
        "Der Kreditbetrag wird gegen den historischen Portfolio-Durchschnitt in einen z-Score "
        "umgerechnet. Der Zinssatz wird gegen den für den jeweiligen Kreditbetrag erwarteten Wert "
        "verglichen (Regression), da Betrag und Zins historisch zusammenhängen. Auf diesen "
        "Features rechnet der Isolation Forest – nicht auf den Rohwerten direkt."
    )
    feature_columns = [c for c in result.columns if c.startswith("Feature_")]
    st.dataframe(
        result[["ContractID", *feature_columns, "Anomaly_Score"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# --- Glossar --------------------------------------------------------------------
with st.expander("📖 Fachbegriffe kurz erklärt"):
    st.markdown(
        "- **Isolation Forest** – ML-Verfahren zur Ausreißererkennung. Es muss nicht vorher "
        "wissen, wie ein Fehler aussieht; es lernt nur, wie *normal* aussieht, und meldet, "
        "was davon abweicht (*unsupervised*).\n"
        "- **Anomaly Score** – Auffälligkeits-Maß je Vertrag. Je niedriger (negativer), desto "
        "auffälliger.\n"
        "- **Sigma (σ) / Z-Score** – Wie viele Standardabweichungen ein Wert vom historischen "
        "Mittel entfernt liegt. *13,9 σ* bedeutet: extrem weit außerhalb des Üblichen.\n"
        "- **Schwellenwert** – Aus der Historie abgeleitete Grenze; ein Vertrag darunter gilt "
        "als Anomalie.\n"
        "- **Perzentil der Historie** – Stellschraube für die Prüfschärfe (der Regler links in "
        "der Seitenleiste).\n"
        "- **Feature Store** – die aufbereiteten Merkmale je Vertrag, auf denen das Modell "
        "rechnet (z. B. Abweichung des Zinssatzes vom für die jeweilige Kredithöhe erwarteten Wert)."
    )
