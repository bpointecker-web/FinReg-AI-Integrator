"""Stellt sicher, dass die Feature-Stores vor den Tests existieren.

Die generierten CSVs liegen bewusst nicht im Repo (siehe .gitignore). Damit
`pytest` auch auf einem frischen Klon ohne manuellen Vorlauf grün läuft, werden
sie hier einmalig aus den Rohdaten erzeugt, falls sie fehlen.
"""

import os

import generate_feature_store
from anomaly_detection import CURRENT_FEATURE_STORE_PATH, HISTORICAL_FEATURE_STORE_PATH

if not (os.path.exists(HISTORICAL_FEATURE_STORE_PATH) and os.path.exists(CURRENT_FEATURE_STORE_PATH)):
    generate_feature_store.main()
