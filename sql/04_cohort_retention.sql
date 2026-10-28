-- UTC calendar dates. Exclude the final potentially partial date.
-- Each retained numerator is restricted to exactly the same eligible cohort as its denominator.
WITH params AS (SELECT CAST(MAX(auth_ts)/86400 AS INTEGER) cutoff_day FROM auth),
 horizons(n) AS (VALUES(1),(7),(30)),
 eligible AS (
 SELECT r.uid,strftime('%Y-%m',r.reg_ts,'unixepoch') cohort,
 CAST(r.reg_ts/86400 AS INTEGER) reg_day,h.n
 FROM reg r CROSS JOIN horizons h CROSS JOIN params p
 WHERE CAST(r.reg_ts/86400 AS INTEGER)+h.n < p.cutoff_day
 ),
 counts AS (
 SELECT e.cohort,e.n,COUNT(DISTINCT e.uid) eligible,
 COUNT(DISTINCT CASE WHEN a.uid IS NOT NULL THEN e.uid END) retained
 FROM eligible e LEFT JOIN auth a ON a.uid=e.uid
 AND a.auth_ts >= (e.reg_day+e.n)*86400 AND a.auth_ts < (e.reg_day+e.n+1)*86400
 GROUP BY e.cohort,e.n
 )
SELECT cohort,n day,eligible,retained,1.0*retained/eligible rate
FROM counts ORDER BY cohort,day;