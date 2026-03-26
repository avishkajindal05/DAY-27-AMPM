# ZeptoFresh – 15-Minute Food & Essentials Delivery
## Case Study Model Answers
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**

---

## (a) Data Quality Diagnosis

### Problem 1 — `delivery_time_mins` = 0 (214 rows)
- **Issue Type:** Data Entry Error / Structural Issue
- **Explanation:** A delivery time of 0 minutes is physically impossible. These records likely represent failed order logging, cancelled orders recorded as delivered, or system timestamp errors.
- **Treatment:** Flag all rows where `delivery_time_mins == 0` for investigation. If no valid delivery occurred, remove these rows from the training dataset. If they represent cancelled orders, create a separate `is_cancelled` flag and exclude them from the late-delivery model.

---

### Problem 2 — `order_value_Rs` = ₹2,95,000 (Extreme Outlier)
- **Issue Type:** Outlier
- **Explanation:** The mean is ₹620 and median is ₹310. A single bakery order worth ₹2.95 lakh is ~476x the median and would severely distort any model trained on this feature. It is almost certainly a data entry error (e.g., extra zeros entered).
- **Treatment:** Apply IQR-based capping: compute Q1, Q3, and IQR for `order_value_Rs`; cap values above `Q3 + 3×IQR`. Alternatively, verify the original order record and correct or remove it. Log-transform `order_value_Rs` before modeling to reduce the influence of extreme values.

---

### Problem 3 — `prep_time_mins` has Negative Values
- **Issue Type:** Data Entry Error / Invalid Value
- **Explanation:** Preparation time cannot be negative. A `prep_time_mins` of −6 means preparation ended before it started — this is logically impossible and indicates a timestamp recording error (e.g., order prep-end timestamp logged before prep-start).
- **Treatment:** Set a validity constraint: `prep_time_mins >= 0`. Replace all negative values with `NaN`, then impute using the median `prep_time_mins` grouped by `order_category` (since prep time varies by food type — Fresh Food will differ from Bakery).

---

### Problem 4 — `customer_rating` has 9,800 Nulls + Values of 0
- **Issue Type:** Missing Values + Invalid Values
- **Explanation:** `customer_rating` is expected on a 1–5 scale. The 9,800 null values represent orders where customers did not submit a rating (Missing Not At Random — customers who had bad experiences may be less likely to rate). Additionally, ratings of 0 are outside the valid range and are likely default/unsubmitted values incorrectly stored as 0.
- **Treatment:**
  - Replace all `customer_rating == 0` with `NaN` (treat as missing).
  - Do **not** impute with mean/median blindly — instead, create a binary flag `rating_given` (1 if rated, 0 if not), which is itself a predictive signal.
  - Impute missing ratings with the **median rating per hub_id** to respect location-level variation.

---

## (b) Distribution Analysis

### What Does Mean > Median Indicate?

When **mean (18.4) > median (14.2)**, the distribution is **right-skewed (positively skewed)**. This means:
- The majority of deliveries complete in 12–15 minutes (close to the median).
- A long tail of delayed deliveries (30–142 minutes) pulls the mean upward.
- The distribution is **not symmetric** — standard assumptions of normality do not hold.

---

### Rough ASCII Histogram of `delivery_time_mins`

```
Frequency
  |
  |  ████
  |  ████
  |  ████ ██
  |  ████ ██
  |  ████ ██ █
  |  ████ ██ █  █
  |  ████ ██ █  █
  |  ████ ██ █  █  .  .
  +--+----+--+--+--+--+--+--> delivery_time_mins
     5   14  20  28  40  60+
        ↑         ↑
      Median     Mean
     (14.2)     (18.4)
```

- **Tall left peak:** Most orders delivered in 12–16 minutes.
- **Right tail:** Small but significant proportion take 30–142 minutes.
- The gap between median and mean (4.2 minutes) confirms strong right skew.

---

### Transformation to Apply Before Modeling

**Apply Log Transformation:** `log_delivery_time = log(delivery_time_mins + 1)`

**Why:**
1. Log transformation compresses the long right tail, making the distribution approximately normal.
2. Most ML algorithms (Linear Regression, Logistic Regression) assume or perform better with normally distributed features.
3. The `+1` ensures no undefined values when `delivery_time_mins = 0` (though those rows should be removed first as per Question a).
4. After log transformation, the relationship between features and target becomes more linear, improving model convergence and coefficient interpretability.

> **Alternative:** If using tree-based models (Random Forest, XGBoost), transformation is less critical as these are invariant to monotonic transformations — but it still helps with regularized linear models used for baseline comparison.

---

## (c) Correlation Interpretation

### What Is Logically Incorrect About the PM's Conclusion?

The product manager's statement — *"Late deliveries cause refunds. So solving delay will eliminate refunds."* — commits two logical fallacies:

1. **Correlation ≠ Causation:** A correlation of r = +0.74 between `delivery_time_mins` and `refund_issued` means these two variables move together, but does **not** prove that delivery delay is the direct cause of refunds.

2. **Reverse Causality / Oversimplification:** Even if delay does contribute to refunds, eliminating delay alone will not eliminate refunds. Refunds can be issued for reasons entirely unrelated to delivery time (wrong items, damaged goods, quality issues).

---

### What Does Correlation Actually Indicate?

Correlation (r = +0.74) indicates:
- A **strong positive linear relationship** between `delivery_time_mins` and `refund_issued`.
- When delivery time is longer, refund probability tends to be higher — and vice versa.
- **Magnitude:** r = 0.74 is strong but not perfect (r = 1.0 would be perfect linear dependence).
- It says nothing about which variable causes the other, or whether both are driven by a third variable.

---

### Two Possible Confounders

| Confounder | How It Explains Both Variables |
|---|---|
| **`rain_flag`** | Rain increases delivery time (r = +0.48 already shown) AND independently increases damage to packaging/food quality, leading to more refunds. Both delay and refunds rise together due to rain — not because delay causes refunds. |
| **`order_category` (e.g., Fresh Food)** | Perishable or temperature-sensitive orders (Fresh Food) are more likely to be delayed (complex prep) AND more likely to result in refunds (spoilage, quality degradation). The product type drives both outcomes simultaneously. |

---

## (d) Bimodal Pattern in Tier-1 Cities

### Operational Reasons for the Bimodal Pattern

The bimodal distribution (peaks at 12–14 min and 28–32 min) in Mumbai and Bangalore likely reflects **two distinct operational sub-populations** within the same dataset:

1. **Peak 1 (12–14 minutes) — Normal Fulfillment:**
   Orders placed during off-peak hours from nearby hubs, with short rider distance, fast prep (Grocery/Bakery), and no adverse weather. These complete within the 15-minute SLA.

2. **Peak 2 (28–32 minutes) — Stressed Fulfillment:**
   Orders during rush hours (e.g., 7–9 AM, 12–2 PM, 7–10 PM) where multiple factors compound: longer `rider_distance_km` due to hub congestion, higher `prep_time_mins` for Fresh Food/meals, traffic in dense Tier-1 city grids, and high concurrent order volume per hub.

   Additional reasons specific to Tier-1 cities:
   - **Hub overflow:** Orders rerouted to a farther hub when the nearest is at capacity.
   - **Complex addresses:** High-rise buildings, gated societies causing last-mile delays.
   - **Rider reallocation:** Riders serving multiple zones simultaneously during peak demand.

---

### Why Must This Be Addressed Before Modeling?

A bimodal distribution violates the **unimodal normality assumption** of many ML algorithms. If treated as a single population:
- Summary statistics (mean, variance) become meaningless — the mean (~21 min) falls in the **valley** between the two peaks, representing almost no real order.
- Feature distributions and relationships differ between the two sub-populations, making it impossible for a single model to learn coherent patterns.

---

### Modeling Mistake if Ignored

If the bimodal pattern is ignored:

> **The model will learn a blurred, averaged decision boundary that is accurate for neither sub-population.**

Specifically:
- The model may treat the 28–32 min peak as outliers or noise and underfit the high-delay segment.
- **Class imbalance within the bimodal groups** will cause the model to be overconfident in predicting the dominant (fast-delivery) pattern.
- Feature importance will be misleading — variables like `order_hour` or `hub_id` that separate the two modes will appear less important than they truly are.
- **Practical consequence:** The late-delivery risk model will systematically underestimate risk for Tier-1 city peak-hour orders — exactly the orders that need intervention most.

**Correct approach:** Segment the data by city tier (or use `city` + `order_hour` as stratification variables) and train separate models, or include `order_hour × city_tier` interaction features to let the model learn separate patterns.

---

## (e) Business Trade-Off: Precision vs. Recall

### Answer: **Recall should be prioritized — but with a minimum Precision floor.**

---

### Reasoning

**Recall** = Of all orders that will actually be late, what fraction does the model catch?

**Precision** = Of all orders the model flags as late-risk, what fraction are actually late?

Kavya's statement defines the trade-off clearly:

| Scenario | Error Type | Business Consequence |
|---|---|---|
| Model misses a truly late order (not flagged) | **False Negative → Low Recall** | Customer receives 30+ min delivery → churn risk increases. No preemptive intervention was triggered. **High customer cost.** |
| Model flags an on-time order as late-risk | **False Positive → Low Precision** | Unnecessary rider reallocation triggered. **Operational cost increase, but no customer harm.** |

Since **customer churn from late delivery is harder and more expensive to recover from than the cost of an extra rider reallocation**, minimizing False Negatives (maximizing Recall) is the right primary objective.

---

### Threshold Recommendation

- Lower the classification threshold (e.g., from 0.5 → 0.3) to increase Recall.
- Set a **minimum Precision floor** (e.g., ≥ 45–50%) to prevent operational costs from becoming unsustainable — flagging every order as late-risk is not viable.
- Use **F-beta score (β > 1)** as the optimization metric, weighting Recall more heavily than Precision.

---

## (f) Advanced Feature Engineering

### Feature 1
```
prep_to_delivery_ratio = prep_time_mins / delivery_time_mins
```
**Rationale:** Captures what fraction of total delivery time was consumed by preparation alone. A high ratio (>0.7) means the kitchen is the bottleneck, not the rider. This directly distinguishes prep-delay from transit-delay, improving the model's ability to identify root cause of lateness.

---

### Feature 2
```
order_complexity_score = items_count × (1 + 0.5 × (order_category == 'Fresh Food'))
```
**Rationale:** Combines item count with order category complexity. Fresh Food orders require more careful handling and have higher prep variance than Grocery. This single feature encodes both volume and complexity, which jointly drive `prep_time_mins` and delay risk.

---

### Feature 3
```
peak_hour_flag = 1 if order_hour in [7,8,9,12,13,19,20,21,22] else 0
```
**Rationale:** Encodes whether the order was placed during known high-traffic windows (breakfast, lunch, dinner rush). Combined with `is_weekend`, this creates a strong contextual signal for hub congestion and rider availability — two of the primary drivers of the 28–32 min delay peak seen in Tier-1 cities.

---

### Bonus Feature
```
effective_speed = rider_distance_km / (delivery_time_mins - prep_time_mins)
```
**Rationale:** Isolates actual rider transit speed by subtracting prep time from total delivery time. Low `effective_speed` indicates traffic congestion or route inefficiency independent of kitchen performance — a cleaner signal for transit-specific delay prediction.

---

*End of Case Study Answers — ZeptoFresh EDA & Late Delivery Risk*
