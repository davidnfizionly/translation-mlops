-- Sentences (lower-cased) found in more than one split, per field (expected 0)
SELECT 'english' AS field, count(*) AS shared_sentences
FROM (SELECT lower(english) FROM pairs GROUP BY 1 HAVING count(DISTINCT split) > 1)
UNION ALL
SELECT 'nuer', count(*)
FROM (SELECT lower(nuer) FROM pairs GROUP BY 1 HAVING count(DISTINCT split) > 1)
UNION ALL
SELECT 'din', count(*)
FROM (SELECT lower(din) FROM pairs GROUP BY 1 HAVING count(DISTINCT split) > 1);
