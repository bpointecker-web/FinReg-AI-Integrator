-- ===============================================================================
-- Script: Feature Engineering für ML-basierte Anomalieerkennung im Meldewesen
-- ===============================================================================

-- 1. SIMULATION DER ROHDATEN (Entspricht Schritt 1 in der Web-App)
-- -------------------------------------------------------------------------------
IF OBJECT_ID('tempdb..#RawRegulatoryData') IS NOT NULL DROP TABLE #RawRegulatoryData;
CREATE TABLE #RawRegulatoryData (
    Vertragsnr VARCHAR(50),
    Kundennummer VARCHAR(50),
    ReportDate DATE,
    Kreditvolumen_EUR DECIMAL(18,2),
    Zinssatz_Prozent DECIMAL(5,4),
    Laufzeit_Monate INT,
    Basel_Risiko_Klasse VARCHAR(10)
);

-- Mock-Daten einfügen (10 Fälle inkl. Vertrag 47110004 als massive Anomalie)
INSERT INTO #RawRegulatoryData VALUES 
('47110001', '8001020', '2026-03-31', 150000.00,  0.0350, 120, 'Low'),
('47110002', '8001021', '2026-03-31', 250000.00,  0.0320, 180, 'Low'),
('47110003', '8001022', '2026-03-31', 80000.00,   0.0510, 60,  'Medium'),
('47110004', '8001023', '2026-03-31', 4500000.00, 0.0150, 12,  'High'), -- < ANOMALIE
('47110005', '8001024', '2026-03-31', 300000.00,  0.0400, 240, 'Low'),
('47110006', '8001025', '2026-03-31', 120000.00,  0.0380, 120, 'Low'),
('47110007', '8001026', '2026-03-31', 450000.00,  0.0290, 240, 'Low'),
('47110008', '8001027', '2026-03-31', 95000.00,   0.0480, 84,  'Medium'),
('47110009', '8001028', '2026-03-31', 210000.00,  0.0340, 180, 'Low'),
('47110010', '8001029', '2026-03-31', 60000.00,   0.0550, 60,  'Medium');

-- 2. FEATURE ENGINEERING (Die Datenaufbereitung für das ML-Modell)
-- -------------------------------------------------------------------------------
WITH AggregatedHistory AS (
    -- Berechnung statistischer Basiswerte über das Portfolio (z.B. 3-Monats-Schnitt)
    SELECT 
        Basel_Risiko_Klasse,
        AVG(Kreditvolumen_EUR) AS Avg_Volumen,
        STDEV(Kreditvolumen_EUR) AS StDev_Volumen
    FROM #RawRegulatoryData
    GROUP BY Basel_Risiko_Klasse
),
FeatureStore AS (
    -- Transformation in maschinenlesbare Features
    SELECT 
        r.Vertragsnr,
        r.Kundennummer,
        
        -- Feature 1: Volumen Abweichung (Z-Score)
        CASE 
            WHEN a.StDev_Volumen = 0 OR a.StDev_Volumen IS NULL THEN 0
            ELSE (r.Kreditvolumen_EUR - a.Avg_Volumen) / a.StDev_Volumen 
        END AS Volumen_Abweichung_3M,
        
        -- Feature 2: Zins-Volumen-Ratio
        (r.Zinssatz_Prozent * 100) / NULLIF(r.Laufzeit_Monate, 0) AS Zins_Laufzeit_Index,
        
        -- Feature 3: Numerisches Encoding der Risikoklasse
        CASE r.Basel_Risiko_Klasse 
            WHEN 'Low' THEN 1 
            WHEN 'Medium' THEN 2 
            WHEN 'High' THEN 3 
            ELSE 0 
        END AS Basel_Risiko_Code

    FROM #RawRegulatoryData r
    LEFT JOIN AggregatedHistory a ON r.Basel_Risiko_Klasse = a.Basel_Risiko_Klasse
)

-- Output für die ML-Pipeline (Entspricht Schritt 2 in der Web-App)
SELECT * FROM FeatureStore;
