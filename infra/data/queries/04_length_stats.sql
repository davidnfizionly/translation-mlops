-- Word counts and character length ratios per split
SELECT split,
       approx_percentile(cardinality(split(english, ' ')), 0.5) AS en_words_p50,
       max(cardinality(split(english, ' '))) AS en_words_max,
       approx_percentile(cardinality(split(din, ' ')), 0.5) AS din_words_p50,
       approx_percentile(cardinality(split(nuer, ' ')), 0.5) AS nuer_words_p50,
       round(min(length(din) * 1.0 / length(english)), 2) AS din_ratio_min,
       round(max(length(din) * 1.0 / length(english)), 2) AS din_ratio_max,
       round(min(length(nuer) * 1.0 / length(english)), 2) AS nuer_ratio_min,
       round(max(length(nuer) * 1.0 / length(english)), 2) AS nuer_ratio_max
FROM pairs
GROUP BY split
ORDER BY split;
