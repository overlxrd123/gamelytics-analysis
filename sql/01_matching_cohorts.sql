-- Numeric ID equality is a candidate linkage only; P3 registration cohort is unknown.
WITH covered AS (
 SELECT strftime('%Y-%m',r.reg_ts,'unixepoch') cohort, COUNT(*) reg_users,
 SUM(CASE WHEN a.user_id IS NOT NULL THEN 1 ELSE 0 END) matched
 FROM reg r LEFT JOIN ab a ON a.user_id=r.uid
 GROUP BY 1
)
SELECT cohort, reg_users, matched, 1.0*matched/reg_users coverage_rate
FROM covered ORDER BY cohort;