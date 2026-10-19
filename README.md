# Credit Card Default Prediction

Predicting which credit card customers will miss their next payment, so a bank can act before the loss happens.

## The problem

When customers stop paying their credit card bills, the bank loses money. If the bank can spot these customers early, it can act: lower the credit limit, contact the customer, or offer a payment plan.

**Main question:** Can we predict which customers will default on their payment next month (October 2005)?

## Data

- Source: [UCI "Default of Credit Card Clients" dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (Kaggle copy), from a bank in Taiwan.
- 30,000 customers, 25 columns. No missing values and no duplicate IDs.
- Profile: credit limit, sex, education, marital status and age.
- Payment history from April to September 2005: payment status, bill amount and amount paid for each month.
- Target: `default.payment.next.month` (1 = defaulted, 0 = paid). About **22%** of customers default (6,636 out of 30,000), so the classes are imbalanced.
- Amounts are in New Taiwan dollars (NTD).

## What the notebook does

1. **Data check.** Looks at shape, types, missing values and duplicate IDs. Renames `PAY_0` to `PAY_1`, because there is no `PAY_1` in the original file.
2. **Target and baseline.** A model that always says "will pay" is right **77.9%** of the time. Any useful model has to beat this, so the notebook focuses on recall and precision for defaulters, not just accuracy.
3. **Cleaning.** Recodes `SEX` to 0 = male, 1 = female. Merges `EDUCATION` code 6 into 5 (both "unknown").
4. **Exploration.** Compares defaulters and non-defaulters by sex, education, marital status and credit limit.
5. **Feature engineering.**
   - `TOTAL PAID`: total amount paid over the six months.
   - `paid_ratio_1_%` to `paid_ratio_5_%`: share of each bill that was paid the following month. If nothing was owed, it counts as 100%.
   - `utilisation_1` to `utilisation_6`: bill divided by credit limit.
   - `months_late`, `utilisation_avg`, `paid_ratio_avg`: summary features to reduce multicollinearity.
6. **Feature selection.** Keeps features with a correlation above |0.1| with the target.
7. **Models.** All use the same 80/20 split, stratified by the target.
   - Logistic Regression: no preparation, with preparation, scaled, and scaled with balanced class weights.
   - KNN (scaled, distance weights), with a search over the number of neighbours.
   - Decision Tree: untuned, tuned with grid search (entropy, max depth 4), and tuned with balanced class weights.
8. **Interpretation.** Coefficients of the scaled, balanced Logistic Regression, and a threshold analysis (true vs false positives from 0 to 1).

## Results (test set: 6,000 customers, 1,327 defaulters)

| Model | Accuracy | Precision (default) | Recall (default) | F1 (default) |
|---|---|---|---|---|
| Logistic Regression (no preparation) | 0.78 | 0.61 | 0.04 | 0.08 |
| Logistic Regression (prepared) | 0.81 | 0.67 | 0.27 | 0.38 |
| Logistic Regression (scaled) | 0.81 | 0.66 | 0.25 | 0.37 |
| **Logistic Regression (scaled, balanced)** | 0.75 | 0.44 | **0.57** | 0.50 |
| KNN (k = 5, distance weights) | 0.79 | 0.54 | 0.38 | 0.45 |
| Decision Tree (untuned) | 0.73 | 0.40 | 0.39 | 0.39 |
| Decision Tree (tuned) | **0.82** | **0.67** | 0.35 | 0.46 |
| **Decision Tree (tuned, balanced)** | 0.78 | 0.50 | 0.55 | **0.52** |

- **Without class weights, every model is too cautious.** It mostly predicts "will pay" and misses most defaulters.
- **Balanced class weights made the biggest difference.** Logistic Regression went from catching 338 defaulters to 753.
- **Logistic Regression (balanced)** catches the most defaulters (57%). **Decision Tree (balanced)** is close (55%), with fewer false alarms and the best F1.
- **KNN** stays around 38% recall whatever k is chosen. It has no class weight option, so it isn't directly comparable with the balanced models.

**Strongest signals** (scaled, balanced Logistic Regression):
- More likely to default: late payment last month (`PAY_1`, +80% odds per standard deviation), late payments in the two months before, and high credit utilisation.
- Less likely to default: paying more in total (`TOTAL PAID`, -27%) and a higher credit limit (`LIMIT_BAL`, -12%).
- The paid ratio coefficients point the wrong way (higher ratio, higher risk). This is caused by its overlap with payment status, so they shouldn't be read on their own.

**Threshold:** for the balanced Logistic Regression, a threshold of about 0.45–0.5 is a good balance. At 0.5 the model catches 753 defaulters with 954 false alarms. A lower threshold (about 0.4) suits cheap actions like a reminder; a higher one (0.6+) suits costly actions like cutting a credit limit.

## Known limitations

- Features were chosen using correlations on all 30,000 rows, including the test set. This is a small leak.
- The number of neighbours for KNN was chosen using the test set, not cross-validation.
- The Decision Tree grid search was scored on accuracy, which isn't the metric that matters here.
- The final feature set still has correlated groups (six payment statuses, four utilisation columns, two paid ratios), so individual coefficients inside these groups are unstable.
- Sex, age and marital status may be restricted in real credit decisions (UK/EU). They are kept here for analysis only.
- The data is from one bank in Taiwan in 2005, so results may not hold elsewhere.

## Suggested next steps

1. Use the summary features (`months_late`, `utilisation_avg`, `paid_ratio_avg`) instead of the correlated groups, and pick features using the training data only.
2. Merge the rare `EDUCATION` (0, 4, 5, 6) and `MARRIAGE` (0, 3) codes into "other" and add them as dummies.
3. Choose KNN's k and the tree settings with cross-validation on the training data, scored on F1 or recall.
4. Add ROC AUC to compare models across all thresholds.
5. Give a cost to a missed defaulter and to a false alarm, and pick the threshold that costs the bank the least.
6. Try Random Forest or XGBoost.

## How to run

```
pip install pandas plotly matplotlib scikit-learn kagglehub
jupyter notebook CreditCardDefault.ipynb
```

The notebook downloads the data from Kaggle with `kagglehub`. It was built in Google Colab.
