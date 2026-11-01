# Early Diabetes Prediction - adult women AND men

Both-sexes version of the diabetes project. It replaces the Pima dataset (women only) with the
100,000-row "Diabetes prediction dataset" (female and male, with HbA1c and blood glucose).

```bash
pip install -r requirements.txt
python train.py --data data/diabetes_prediction_dataset.csv   # ~2-3 min, writes ./models and ./outputs
streamlit run app.py
```

## React + Node dashboard

The project also includes a React/Vite dashboard and an Express API. The Node API keeps
the trained Python preprocessing and model artifacts as the source of truth by invoking
`predict.py` for each assessment.

```bash
npm install
npm run install:all
npm run dev
```

Open `http://127.0.0.1:5173`. The API runs on `http://127.0.0.1:3001`. Login is a
local demo flow: use any email address and a password with at least six characters.
This authentication is not intended for production use.

## Two models, because the labs change everything

| | Inputs | Test ROC-AUC | Recall | Precision | Use |
|---|---|---|---|---|---|
| **screening** | sex, age, BMI, hypertension, heart disease, smoking | ~0.80 | 85% | ~20% | "Who should get a blood test?" - a real early screen |
| **full** | screening inputs + HbA1c + blood glucose | ~0.975 | 85% | ~63% | Close to confirming a diagnosis |

In this dataset the label follows HbA1c / glucose thresholds almost exactly (e.g. every row with
HbA1c above 6.6 is positive), so a model that sees the labs is closer to *re-applying the diagnostic
definition* than to *predicting early*. The screening model is the honest early-prediction model; the
app picks one automatically (toggle "I have recent blood test results").

## What changed from the Pima version

| Topic | Pima version | Both-sexes version |
|---|---|---|
| Population | 768 women (Pima heritage) | ~79k adults, 60% women / 40% men |
| Biological zeros | zeros in 5 columns -> NaN | none; instead the **BMI placeholder 27.32** (24% of rows, an upstream median-fill marker) -> NaN |
| Sex | n/a | `is_male` feature; tested separately per sex |
| Smoking | n/a | 6 raw labels -> never / former / current / unknown |
| Features | Pregnancies, SkinThickness, Insulin, pedigree... | hypertension, heart disease, smoking, HbA1c, glucose |
| Engineered | BMI/glucose ratio, HOMA proxy | Age_Group, BMI_Category, Age_BMI_Risk, Cardiometabolic_Burden, HbA1c/Glucose categories, glucose-vs-HbA1c gap |

## Dataset cleaning (printed by `train.py`)

100,000 rows -> drop 18 `Other` gender rows (all non-diabetic, too few to learn from) -> keep adults
18+ (17% of rows were children; type-2 prediction is not meaningful for them) -> remove 3,300+ exact
duplicates **before** the split so no patient is in both train and test -> **79,444 rows, 10.6% diabetic**.
The name / patient-ID columns in the source file are dropped and not saved.

## Design decisions

* **No leakage:** the split comes first; medians, IQR fences and scaler statistics are learned from the
  training fold only (the whole chain is one sklearn `Pipeline`, refit in every CV fold).
* **Threshold tuned on out-of-fold training predictions**, never the test set. It is the highest-precision
  threshold with recall >= 85%. Test recall at that threshold is ~85% (95% CI ~83-87%).
* **Random Forest without `class_weight`:** reweighting inflates predicted probabilities, and the app shows
  probabilities to users. Recall is reached through the threshold instead.
* **Same threshold for women and men**; `train.py` prints recall / precision / ROC-AUC per sex.
  Recall is a little lower for women (~0.83 vs ~0.88) - worth knowing if you act on it.
* **No "BMI was missing" indicator:** rows with the placeholder BMI have a different diabetes rate for
  data-collection reasons, not biology.
* **Explanations** group derived features under the measurement they come from (e.g. Age_Group -> Age).

## Limitations - please read

* **Unknown provenance.** The dataset's origin, collection method and label definition are not documented.
  The labs-to-label relationship above is unusually clean, which is a warning sign that the data may be
  simulated or rule-labelled. I could not verify either way. Do not assume it reflects a real population.
* **Not validated on any external data**, and no demographic information (ethnicity, country) is available.
* **Screening model precision is ~20%**: about 4 in 5 flagged people do not have diabetes. That is the
  cost of 85% recall with weak predictors, and a flag only ever means "get a blood test".
* Blood glucose in the data has no stated context (fasting vs random), so the app's glucose guidance
  presents both.
* Educational tool, **not a medical device**.
