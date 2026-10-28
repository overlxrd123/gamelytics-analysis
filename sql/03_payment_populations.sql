-- AB_all is the primary experiment sample; matched subset is not the full player population.
WITH populations AS (
 SELECT 'AB_all' population,a.* FROM ab a
 UNION ALL
 SELECT CASE WHEN r.uid IS NULL THEN 'P3_unknown' ELSE 'P2_candidate_matched' END,a.*
 FROM ab a LEFT JOIN reg r ON r.uid=a.user_id
)
SELECT population,COUNT(*) users,SUM(CASE WHEN revenue>0 THEN 1 ELSE 0 END) payers,
 SUM(revenue) revenue,AVG(CASE WHEN revenue>0 THEN 1.0 ELSE 0.0 END) payer_rate,
 AVG(revenue*1.0) arpu,
 1.0*SUM(revenue)/NULLIF(SUM(CASE WHEN revenue>0 THEN 1 ELSE 0 END),0) arppu,
 SUM(CASE WHEN revenue>10000 THEN 1 ELSE 0 END) whale_users,
 1.0*SUM(CASE WHEN revenue>10000 THEN revenue ELSE 0 END)/NULLIF(SUM(revenue),0) whale_revenue_share
FROM populations GROUP BY population ORDER BY population;