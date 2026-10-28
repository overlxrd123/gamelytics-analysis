-- No assumption that balance proves randomization. SRM is computed separately under assumed 1:1.
SELECT a.testgroup,COUNT(*) users,
 SUM(CASE WHEN r.uid IS NOT NULL THEN 1 ELSE 0 END) matched,
 SUM(CASE WHEN r.uid IS NULL THEN 1 ELSE 0 END) unmatched,
 SUM(CASE WHEN a.revenue>0 THEN 1 ELSE 0 END) payers,
 SUM(a.revenue) revenue
FROM ab a LEFT JOIN reg r ON r.uid=a.user_id
GROUP BY a.testgroup ORDER BY a.testgroup;