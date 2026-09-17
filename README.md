# Predicting UK Rail Journey Disruptions (Python / scikit-learn)

🚧 **Work in progress** — this project is being built step-by-step to
predict whether a UK rail journey will be delayed or cancelled, using
only information known at the time of booking.

## Status

- [x] Data loaded and explored
- [x] Target variable defined (disrupted vs. on-time)
- [x] Booking lead time, day-of-week, and duration features engineered
- [x] Fixed a data quality issue: blank `Railcard` values were being
      wrongly dropped as missing data, when they actually mean "no
      railcard used" — fixed by filling with an explicit label
- [ ] Train/test split
- [ ] Baseline model (Logistic Regression)
- [ ] Random Forest model + evaluation
- [ ] Feature importance analysis
- [ ] Final write-up with results

## Dataset

[UK Train Rides](https://www.kaggle.com/datasets/farheenshaukat/uk-train-rides)
(Kaggle) — 31,653 UK rail journeys, Jan–Apr 2024.

## Why this project

Built as a companion to an sql analysis made on of the
same dataset — that project answers "what happened," this one asks
"can we predict what's about to happen."

## Running it so far

```bash
pip install -r requirements.txt
python3 walkthrough.py   # run cell by cell in VS Code, or top to bottom
```

More to come as each stage is completed.