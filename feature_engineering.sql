-- ===============================================================================
-- Script: Feature Engineering für ML-basierte Anomalieerkennung im Meldewesen
-- ===============================================================================

-- 1. SIMULATION DER ROHDATEN (Nur für den Showroom, in echt deine DWH-Tabellen)
-- -------------------------------------------------------------------------------
IF OBJECT_ID('tempdb..#RawRegulatoryData') IS NOT NULL DROP TABLE #RawRegulatoryData;
CREATE TABLE #RawRegulatoryData (
    ContractID VARCHAR(50),
    CustomerID VARCHAR(50),
    ReportDate DATE,
    LoanAmount_EUR DECIMAL(18,2),
    InterestRate DECIMAL(5,4),
    Duration_Months INT,
    Basel_Risk_Class VARCHAR(10)
);

-- Mock-Daten einfügen (Normale Fälle + eine versteckte Anomalie)
INSERT INTO #RawRegulatoryData VALUES 
('C-1001', 'K-99', '2026-03-31', 150000.00, 0.0350, 120, 'Low'),
('C-1002', 'K-12', '2026-03-31', 250000.00, 0.0320, 180, 'Low'),
('C-1003', 'K-45', '2026-03-31', 80000.00,  0.0510, 60,  'Medium'),
('C-1004', 'K-88', '2026-03-31', 4500000.00,0.0150, 12,  'High'), -- < ANOMALIE: Extrem hohes Volumen, kurzer Kredit, paradoxes (High) Risiko bei niedrigem Zins
('C-1005', 'K-33', '2026-03-31', 300000.00, 0.0400, 240, 'Low');

-- 2. FEATURE ENGINEERING (Die eigentliche Datenaufbereitung für die KI)
-- -------------------------------------------------------------------------------
WITH AggregatedHistory AS (
    -- Wir berechnen statistische Basiswerte über das gesamte Portfolio als Benchmark
    SELECT 
        Basel_Risk_Class,
        AVG(LoanAmount_EUR) AS Avg_Amount_Per_RiskClass,
        AVG(InterestRate) AS Avg_Interest_Per_RiskClass,
        STDEV(LoanAmount_EUR) AS StDev_Amount_Per_RiskClass
    FROM #RawRegulatoryData
    GROUP BY Basel_Risk_Class
),
FeatureStore AS (
    -- Wir generieren die Features für das ML-Modell (Abweichungen vom Durchschnitt)
    SELECT 
        r.ContractID,
        r.LoanAmount_EUR,
        r.InterestRate,
        r.Duration_Months,
        -- Feature 1: Wie weit weicht das Volumen vom Durchschnitt dieser Risikoklasse ab? (Z-Score ähnlich)
        CASE 
            WHEN a.StDev_Amount_Per_RiskClass = 0 OR a.StDev_Amount_Per_RiskClass IS NULL THEN 0
            ELSE (r.LoanAmount_EUR - a.Avg_Amount_Per_RiskClass) / a.StDev_Amount_Per_RiskClass 
        END AS Feature_AmountDeviation_Score,
        
        -- Feature 2: Zins-Volumen-Ratio (Oft ein starker Indikator für fehlerhafte Eingaben)
        (r.InterestRate * 100) / NULLIF(r.Duration_Months, 0) AS Feature_InterestDuration_Ratio,
        
        -- Feature 3: Risikoklasse numerisch kodieren (ML-Modelle brauchen Zahlen)
        CASE r.Basel_Risk_Class 
            WHEN 'Low' THEN 1 
            WHEN 'Medium' THEN 2 
            WHEN 'High' THEN 3 
            ELSE 0 
        END AS Feature_RiskClass_Encoded

    FROM #RawRegulatoryData r
    LEFT JOIN AggregatedHistory a ON r.Basel_Risk_Class = a.Basel_Risk_Class
)

-- Export-Ansicht für das Python-Skript (In der Praxis ein BCP Export oder direkter DB-Read)
SELECT * FROM FeatureStore;