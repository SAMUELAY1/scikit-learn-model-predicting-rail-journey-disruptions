# Predicting UK Rail Journey Disruptions (Python / scikit-learn)

A classification model that predicts whether a UK rail journey will be
**delayed or cancelled**, using only information known at the time of
booking. Built as a companion to my [https://github.com/SAMUELAY1/RAILWAY--SQL-ANALYSIS-UK]
of the same dataset: that project answers "what happened", this one asks
"can we predict what's about to happen?"

## Dataset

[UK Train Rides](https://www.kaggle.com/datasets/farheenshaukat/uk-train-rides)
(Kaggle): 31,653 UK rail journeys, Jan–Apr 2024. About 13% of journeys were delayed or cancelled.

## Approach

- **Target:** disrupted (Delayed or Cancelled) vs. on time
- **Features:** stations, ticket class/type, railcard, purchase type, payment
  method, price, booking lead time, day of week, month, and scheduled journey duration
- **Avoiding data leakage:** `Reason for Delay`, `Actual Arrival Time` and
  `Refund Request` are excluded because they are only known after a journey goes wrong
- **Data quality fix:** blank `Railcard` values mean "no railcard", not missing data. Dropping them would have discarded about two-thirds of the rows, so they are filled with an explicit `"None"` category
- **Class imbalance:** both models use `class_weight="balanced"`, and
  evaluation uses an 80/20 stratified split

## Results

Evaluated on a held-out test set of 6,331 journeys (834 disrupted).

| Model | ROC-AUC | Accuracy | Disrupted precision | Disrupted recall | Disrupted F1 |
| Logistic Regression (baseline) | 0.742 | 0.79 | 0.32 | 0.57 | 0.41 |
| Random Forest | **0.831** | 0.87 | 0.50 | 0.56 | 0.53 |

**Random Forest confusion matrix**

| | Predicted on time | Predicted disrupted |
| **Actual on time** | 5,028 | 469 |
| **Actual disrupted** | 370 | 464 |

The Random Forest caught 464 of the 834 actual disruptions (56%) and missed 370. When it flags a journey as disrupted it is right about half the time (464 of 933 flags), and it raised 469 false alarms.

**Why accuracy isn't the headline metric:** a model that predicted "on time" for every journey would score about 86.8% accuracy while catching zero disruptions. The Random Forest's 87% accuracy alone is therefore not impressive.ROC-AUC (0.831 vs. 0.5 for guessing) and recall on the disrupted class are the meaningful measures.

## Feature importance (Random Forest)

| Feature | Importance |
| Arrival destination: London Euston | 0.134 |
| Price | 0.123 |
| Scheduled journey duration | 0.089 |
| Purchase type: online | 0.066 |
| Purchase type: station | 0.062 |
| Payment method: debit card | 0.056 |

**Caveats:** importance shows what the model *used*, not what *causes* disruption. Payment method and purchase type ranking highly is surprising, since
they have no obvious operational link to punctuality. This suggests the model maybe picking up quirks of this dataset rather than real drivers of delay. Random
Forest importance can also favour continuous features like price. These findings are a starting point to investigate, not conclusions.

## Limitations and next steps

- Add weather data, since weather was a leading delay reason in the SQL analysis
- Tune hyperparameters (grid or random search); defaults were used here
- Try gradient boosting (XGBoost / LightGBM)
- Try resampling (e.g. SMOTE) as an alternative to class weighting
- Check feature importance with permutation importance, which is less biased
  than the default measure

## Running it

```bash
pip install -r requirements.txt
python3 walkthrough.py   # run cell by cell in VS Code (Python extension), or top to bottom
```