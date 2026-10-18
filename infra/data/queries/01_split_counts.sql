-- Rows and distinct row ids per split
SELECT split, count(*) AS rows, count(DISTINCT row_id) AS distinct_row_ids
FROM pairs
GROUP BY split
ORDER BY split;
