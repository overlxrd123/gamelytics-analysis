-- Unknown behavior remains NULL. This is NOT proof of common user identity.
WITH params AS (SELECT CAST(MAX(auth_ts)/86400 AS INTEGER) cutoff_day FROM auth),
 behavior AS (
 SELECT a.uid,COUNT(*) login_cnt,MAX(CAST(a.auth_ts/86400 AS INTEGER)) last_day,
 COUNT(DISTINCT CASE WHEN a.auth_ts >= (p.cutoff_day-30)*86400
 THEN CAST(a.auth_ts/86400 AS INTEGER) END) recent_active_days
 FROM auth a CROSS JOIN params p WHERE a.auth_ts < p.cutoff_day*86400 GROUP BY a.uid
 )
SELECT a.user_id,a.revenue,a.testgroup,
 CASE WHEN r.uid IS NOT NULL THEN 1 ELSE 0 END matched,
 CASE WHEN r.uid IS NOT NULL THEN COALESCE(b.login_cnt,0) END login_cnt,
 CASE WHEN r.uid IS NOT NULL THEN COALESCE(b.recent_active_days,0) END recent_active_days,
 CASE WHEN r.uid IS NOT NULL THEN p.cutoff_day-b.last_day END recency,
 CASE WHEN r.uid IS NOT NULL THEN p.cutoff_day-CAST(r.reg_ts/86400 AS INTEGER) END tenure_days
FROM ab a CROSS JOIN params p LEFT JOIN reg r ON r.uid=a.user_id AND r.reg_ts<p.cutoff_day*86400
LEFT JOIN behavior b ON b.uid=r.uid ORDER BY a.user_id;