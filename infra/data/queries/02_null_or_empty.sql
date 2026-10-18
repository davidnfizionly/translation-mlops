-- Rows with a missing or blank field, per split (expected 0)
SELECT split,
       sum(CASE WHEN english IS NULL OR trim(english) = '' THEN 1 ELSE 0 END) AS english_empty,
       sum(CASE WHEN nuer IS NULL OR trim(nuer) = '' THEN 1 ELSE 0 END) AS nuer_empty,
       sum(CASE WHEN din IS NULL OR trim(din) = '' THEN 1 ELSE 0 END) AS din_empty
FROM pairs
GROUP BY split
ORDER BY split;
