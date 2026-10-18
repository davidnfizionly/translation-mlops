-- Rows where a target repeats the same three-word sequence three times in a row.
-- Cleaning removes these unless the English repeats too, so a small count can remain.
SELECT split,
       sum(CASE WHEN regexp_like(din, '(\S+ \S+ \S+)( \1){2}') THEN 1 ELSE 0 END) AS din_loops,
       sum(CASE WHEN regexp_like(nuer, '(\S+ \S+ \S+)( \1){2}') THEN 1 ELSE 0 END) AS nuer_loops
FROM pairs
GROUP BY split
ORDER BY split;
