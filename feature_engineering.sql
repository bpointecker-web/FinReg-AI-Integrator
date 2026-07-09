-- ===============================================================================
-- Script: Feature Engineering für ML-basierte Anomalieerkennung im Meldewesen
-- ===============================================================================
-- Hinweis: Dieses Skript ist die fachliche Referenzimplementierung (T-SQL/DWH).
-- Die lokal ausführbare, äquivalente Umsetzung (für Demo/Tests ohne DB-Verbindung)
-- liegt in generate_feature_store.py und bildet exakt dieselbe Logik und
-- Mock-Daten ab.
-- ===============================================================================

-- 1. SIMULATION DER ROHDATEN (Nur für den Showroom, in echt deine DWH-Tabellen)
-- -------------------------------------------------------------------------------
-- WICHTIG: Der Benchmark (Schritt 2) wird ausschließlich aus #HistoricalBenchmarkData
-- berechnet, NIEMALS aus der Periode, die gerade geprüft wird (#CurrentPeriodData).
-- Andernfalls würde ein Ausreißer die Streuung des Vergleichsmaßstabs selbst
-- aufblähen und sich damit unauffällig machen.
-- Hinweis: InterestRate ist bewusst weit dimensioniert (DECIMAL(12,4)), damit auch
-- offensichtlich fehlerhafte Defaultwerte (z.B. 999999) unverfälscht ankommen –
-- genau solche Werte soll die Prüfung ja fangen.
IF OBJECT_ID('tempdb..#HistoricalBenchmarkData') IS NOT NULL DROP TABLE #HistoricalBenchmarkData;
CREATE TABLE #HistoricalBenchmarkData (
    ContractID VARCHAR(50),
    ReportDate DATE,
    LoanAmount_EUR DECIMAL(18,2),
    InterestRate DECIMAL(12,4),
    Duration_Months INT
);

-- Validierte Meldungen aus abgeschlossenen Vorperioden (das "gelernte Normal").
INSERT INTO #HistoricalBenchmarkData VALUES
('00000100841', '2025-12-31', 100000.00, 0.0360, 110),
('00000100842', '2025-12-31', 130000.00, 0.0340, 130),
('00000100843', '2025-12-31', 150000.00, 0.0355, 100),
('00000100844', '2025-12-31', 180000.00, 0.0330, 150),
('00000100845', '2025-12-31', 200000.00, 0.0345, 160),
('00000100846', '2025-12-31', 220000.00, 0.0325, 200),
('00000100847', '2025-12-31', 260000.00, 0.0310, 220),
('00000100848', '2025-12-31', 300000.00, 0.0300, 240),
('00000100849', '2025-12-31', 60000.00,  0.0500, 55),
('00000100850', '2025-12-31', 75000.00,  0.0490, 65),
('00000100851', '2025-12-31', 90000.00,  0.0480, 70),
('00000100852', '2025-12-31', 110000.00, 0.0470, 80),
('00000100853', '2025-12-31', 140000.00, 0.0460, 90),
('00000100854', '2025-12-31', 170000.00, 0.0450, 95),
('00000100855', '2025-12-31', 700000.00, 0.0220, 24),
('00000100856', '2025-12-31', 850000.00, 0.0200, 30),
('00000100857', '2025-12-31', 950000.00, 0.0210, 36),
('00000100858', '2025-12-31', 1100000.00,0.0190, 40),
('00000100859', '2025-12-31', 1400000.00,0.0180, 48);

IF OBJECT_ID('tempdb..#CurrentPeriodData') IS NOT NULL DROP TABLE #CurrentPeriodData;
CREATE TABLE #CurrentPeriodData (
    ContractID VARCHAR(50),
    CustomerID VARCHAR(50),
    ReportDate DATE,
    LoanAmount_EUR DECIMAL(18,2),
    InterestRate DECIMAL(12,4),
    Duration_Months INT
);

-- Die aktuell zu meldende Periode (wird gegen den Benchmark oben geprüft).
INSERT INTO #CurrentPeriodData VALUES
('00000151721', 'K-99', '2026-03-31', 150000.00, 0.0350, 120),
('00000151722', 'K-12', '2026-03-31', 250000.00, 0.0320, 180),
('00000151723', 'K-45', '2026-03-31', 200000.00, 999999.0000, 120), -- < ANOMALIE: Zins-Defaultwert 999999 (Platzhalter aus Vorsystem), Betrag/Laufzeit normal
('00000151724', 'K-88', '2026-03-31', 80000.00,  0.0510, 60),
('00000151725', 'K-33', '2026-03-31', 300000.00, 0.0400, 240),
('00000151726', 'K-21', '2026-03-31', 140000.00, 0.0360, 100),
('00000151727', 'K-56', '2026-03-31', 200000.00, 0.0330, 150),
('00000151728', 'K-77', '2026-03-31', 95000.00,  0.0480, 70),
('00000151729', 'K-64', '2026-03-31', 105000.00, 0.0460, 80),
('00000151730', 'K-19', '2026-03-31', 170000.00, 0.0370, 130);

-- 2. BENCHMARK AUS DER HISTORIE
-- -------------------------------------------------------------------------------
-- WICHTIG: Kreditbetrag und Zinssatz hängen historisch stark zusammen (kleine
-- Kredite haben höhere Zinsen als große). Der Zins-Normalbereich wird daher NICHT
-- als flacher Portfolio-Durchschnitt berechnet (das würde einen für seine Größe
-- unauffälligen Kredit fälschlich als Ausreißer werten, oder umgekehrt einen
-- echten Ausreißer übersehen), sondern per linearer Regression "Zins in
-- Abhängigkeit vom Kreditbetrag". T-SQL kennt keine eingebauten
-- REGR_SLOPE/REGR_INTERCEPT-Funktionen, daher die Methode der kleinsten Quadrate
-- von Hand aus den Summenformeln berechnet.
WITH Aggregates AS (
    SELECT
        CAST(COUNT(*) AS FLOAT)                        AS N,
        SUM(LoanAmount_EUR)                             AS Sum_X,
        SUM(InterestRate)                               AS Sum_Y,
        SUM(LoanAmount_EUR * InterestRate)              AS Sum_XY,
        SUM(LoanAmount_EUR * LoanAmount_EUR)            AS Sum_XX,
        AVG(LoanAmount_EUR)                             AS Avg_Amount,
        STDEV(LoanAmount_EUR)                            AS StDev_Amount
    FROM #HistoricalBenchmarkData
),
RegressionParams AS (
    SELECT
        N, Avg_Amount, StDev_Amount,
        (N * Sum_XY - Sum_X * Sum_Y) / (N * Sum_XX - Sum_X * Sum_X) AS Slope,
        (Sum_Y - ((N * Sum_XY - Sum_X * Sum_Y) / (N * Sum_XX - Sum_X * Sum_X)) * Sum_X) / N AS Intercept
    FROM Aggregates
),
Residuals AS (
    -- Abweichung jedes historischen Vertrags vom für seinen Betrag erwarteten Zins
    SELECT
        p.N,
        POWER(h.InterestRate - (p.Intercept + p.Slope * h.LoanAmount_EUR), 2) AS Squared_Residual
    FROM #HistoricalBenchmarkData h
    CROSS JOIN RegressionParams p
),
ResidualStats AS (
    -- ddof=2, da zwei Regressionsparameter (Steigung, Achsenabschnitt) geschätzt wurden
    SELECT SQRT(SUM(Squared_Residual) / (MAX(N) - 2)) AS StDev_InterestResidual
    FROM Residuals
),
Benchmark AS (
    SELECT r.Avg_Amount, r.StDev_Amount, r.Slope, r.Intercept, rs.StDev_InterestResidual
    FROM RegressionParams r
    CROSS JOIN ResidualStats rs
),

-- 3. FEATURE ENGINEERING FÜR DIE AKTUELLE PERIODE
-- -------------------------------------------------------------------------------
FeatureStore AS (
    SELECT
        c.ContractID,
        -- Roh-Vertragsdaten mitführen, damit das Dashboard die echten Werte zeigen kann.
        c.LoanAmount_EUR,
        c.InterestRate,
        c.Duration_Months,
        -- Feature 1: Wie weit weicht das Volumen vom historischen Portfolio-Durchschnitt
        -- ab (Z-Score gegen die Historie, nicht gegen sich selbst)?
        CASE
            WHEN b.StDev_Amount = 0 OR b.StDev_Amount IS NULL THEN 0
            ELSE (c.LoanAmount_EUR - b.Avg_Amount) / b.StDev_Amount
        END AS Feature_AmountDeviation_Score,

        -- Feature 2: Wie weit weicht der Zinssatz vom für DIESEN Kreditbetrag
        -- erwarteten Zins ab (Regressions-Residuum, siehe oben)?
        CASE
            WHEN b.StDev_InterestResidual = 0 OR b.StDev_InterestResidual IS NULL THEN 0
            ELSE (c.InterestRate - (b.Intercept + b.Slope * c.LoanAmount_EUR)) / b.StDev_InterestResidual
        END AS Feature_InterestDeviation_Score

    FROM #CurrentPeriodData c
    CROSS JOIN Benchmark b
)

-- Export-Ansicht für das Python-Skript (in der Praxis ein BCP-Export oder direkter DB-Read).
-- Roh-Vertragsdaten UND abgeleitete Features werden exportiert.
SELECT
    ContractID,
    LoanAmount_EUR,
    InterestRate,
    Duration_Months,
    Feature_AmountDeviation_Score,
    Feature_InterestDeviation_Score
FROM FeatureStore;
