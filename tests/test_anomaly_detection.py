import pytest

from anomaly_detection import (
    CURRENT_FEATURE_STORE_PATH,
    HISTORICAL_FEATURE_STORE_PATH,
    RAW_DISPLAY_COLUMNS,
    load_feature_store,
    score_current_period,
)

# Bewusst eingebaute Anomalie: Vertrag mit einem unmöglichen Zinssatz (Defaultwert 999999).
ANOMALY_ID = "00000151723"


def test_known_anomaly_is_flagged():
    # Integrationstest gegen die im Projekt mitgelieferten Feature-Stores
    # (erzeugt von generate_feature_store.py).
    historical_df = load_feature_store(HISTORICAL_FEATURE_STORE_PATH)
    current_df = load_feature_store(CURRENT_FEATURE_STORE_PATH)

    result = score_current_period(historical_df, current_df)

    flagged = set(result.loc[result["Is_Anomaly"], "ContractID"])
    assert flagged == {ANOMALY_ID}
    # Der Ausreißer muss den schlechtesten (niedrigsten) Score haben und daher zuerst stehen
    assert result.iloc[0]["ContractID"] == ANOMALY_ID
    # Die Begründung nennt das betroffene Bankenattribut im Klartext
    assert "Zinssatz" in result.iloc[0]["Reason"]


def test_contract_ids_keep_leading_zeros_and_raw_columns_present():
    # Führende Nullen der nullgepolsterten Vertragsnummern dürfen nicht verloren gehen
    # (sonst würde 00000151723 zu 151723), und die Roh-Vertragsdaten müssen für die
    # Anzeige mitgeliefert werden.
    current_df = load_feature_store(CURRENT_FEATURE_STORE_PATH)

    # IDs müssen Strings mit erhaltenen führenden Nullen sein (nicht zu int kollabiert)
    assert all(isinstance(cid, str) and cid.startswith("0") for cid in current_df["ContractID"])
    assert ANOMALY_ID in set(current_df["ContractID"])
    for column in RAW_DISPLAY_COLUMNS:
        assert column in current_df.columns


def test_load_feature_store_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_feature_store("data/does_not_exist.csv")


def test_load_feature_store_missing_columns_raises_value_error(tmp_path):
    broken_csv = tmp_path / "broken.csv"
    broken_csv.write_text("ContractID,Feature_AmountDeviation_Score\n00000000001,0.1\n")

    with pytest.raises(ValueError, match="fehlen die Spalten"):
        load_feature_store(str(broken_csv))
