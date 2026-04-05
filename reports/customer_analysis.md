# Customer Behavior Analysis Report
*Blys AI Engineer Technical Assessment — Section 1*

---

## 1. Dataset Overview
| Metric | Value |
|--------|-------|
| Total customers | 20,000 |
| Date range | 2024-01-01 to 2024-04-29 |
| Features | Booking Frequency, Avg Spending, Sentiment Score, Days Inactive |

---

## 2. Data Preprocessing
- **Missing values**: 0 remaining after handling
- **Days Inactive**: derived from `Last_Activity` relative to most recent date
- **Sentiment** (VADER compound score):
  - Positive: 11,935 | Neutral: 6,049 | Negative: 2,016
- **Normalization**: MinMaxScaler on separate cluster matrix; originals preserved

---

## 3. Customer Segmentation (K-Means, k=3)
| Segment              |   Booking_Frequency |   Avg_Spending |   Sentiment_Score |   Days_Inactive |   Customer_Count |
|:---------------------|--------------------:|---------------:|------------------:|----------------:|-----------------:|
| At-Risk Customers    |                3.99 |         273.13 |              0.54 |           60.15 |             6005 |
| Casual Browsers      |                7.54 |         276.18 |             -0.08 |           59    |             8065 |
| High-Value Customers |               10.99 |         274.47 |              0.54 |           60.06 |             5930 |

*Plots: `reports/customer_segments.png`, `reports/elbow_silhouette.png`*

---

## 4. Key Insights

### 4.1 High-Value Customers
- **Count**: 863 (4.3% of base)
- **Criteria**: Top-25% spending (>= $389) AND top-25% frequency AND positive sentiment
- **Avg spending**: $442  |  **Avg frequency**: 12.5 bookings
- **Top services**: Hair Spa, Wellness Package, Body Scrub

**Retention Strategies:**
1. **Loyalty program** — tiered rewards (points per booking, free upgrade at milestones)
2. **Early access** — priority booking for new therapists and premium packages
3. **Personalised offers** — curated discounts on most-booked services
4. **Dedicated account manager** — proactive outreach before typical booking cycle

### 4.2 At-Risk Customers (Churn)
- **Count**: 2,401 (12.0% of base)
- **Criteria**: Top-33% inactivity (>= 80 days) AND (negative sentiment OR bottom-25% frequency)
- **Avg days inactive**: 100  |  **Avg sentiment**: 0.173
- **Top services**: Body Scrub, Facial, Hair Spa

**Re-engagement Tactics:**
1. **Win-back discount** — 15-20% off next booking at 45 days inactivity
2. **Push/SMS reminder** — personalised, references last service booked
3. **Feedback survey** — understand dissatisfaction, offer resolution
4. **New service intro** — highlight untried services to expand basket

---

## 5. Visualisations
| File | Description |
|------|-------------|
| `reports/sentiment_distribution.png` | Sentiment score histogram |
| `reports/elbow_silhouette.png` | Elbow + silhouette plots |
| `reports/customer_segments.png` | PCA cluster scatter + segment sizes |
| `reports/insights.png` | High-value vs churn-risk + service preference |
