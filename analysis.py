"""
Migraine — BEST project: classification + evidence synthesis + design
=====================================================================
Two real data sources, no augmentation, honest evaluation:

PART A — Real patient data (400 physician-diagnosed patients)
  data/migraine.csv — Hospital Materno Infantil de Soledad, Colombia,
  2013; F1000Research / Code Ocean; Kaggle "Migraine Dataset".
  400 rows x 24 columns; 23 features + target Type (7 migraine types).
  Task: classify migraine type. Models: Logistic Regression,
  Random Forest, Gradient Boosting. Stratified 80/20 split
  (320 train / 80 test, random_state=42), 5-fold CV.

PART B — Evidence from the 10 papers (data/papers.csv)
  Published numbers copied exactly from
  ~/workspace/research_notes/migraine-research-papers-20261118-1348/report.md
  Sample sizes, screened prevalence, trigger rates, ML AUCs,
  treatment responder rates, misdiagnosis rates. Analysed and
  compared here — not just listed.

PART C — One idea from EACH paper, turned into project design
  See IDEA_MAP below and outputs/idea_map.csv. The ideas are:
  forecasting (Rabany/Faisal), student survey design (Samir/Almomani),
  missed-diagnosis flags (Angerhoefer/Suwanlaong), trigger awareness
  (Elmazny), real-world treatment evidence (Barbanti/Cardona-Roca),
  and the honesty lesson (Khan). A student survey template built
  from those ideas is in data/survey_template.csv.

HONESTY NOTE — Khan et al. (2024) reported DNN 99.66% on this same
400-record dataset after augmenting it to 1,447 records. That figure
is NOT claimed here: synthetic near-duplicates can leak across
train/test splits and inflate accuracy, there was no external
validation, and the paper itself is internally inconsistent
(Random Forest 98.50% in the abstract vs 88.50% in Table 3).
This project reports the honest 88-94% range.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------------
# IDEA MAP — one idea from each of the 10 papers
# ------------------------------------------------------------------
IDEA_MAP = [
    ("Rabany et al. 2026", "Forecasting, not just classifying: 30-day headache history (37.64% of model performance) beat prodrome (11.13%).",
     "ML evaluation reports AUC-style honesty and a time/history lesson; the project design (REPORT.md §4.3) proposes a 4-8 week daily diary so a future forecasting model has the history it needs. No time-series is faked from the 400 cross-sectional records."),
    ("Faisal et al. 2026", "How you model time matters: time-series AUC 0.84 vs decision tree 0.59 and foundation model 0.55.",
     "Model comparison in Part A is head-to-head on the same split/CV, like Faisal's comparison — and the survey template (Q15) collects the daily data a time-series model would need. Sensitivity/recall is reported per class (REPORT.md §7.2), so a good headline accuracy cannot hide missed cases."),
    ("Samir et al. 2025", "Student survey design: validated screens (MS-Q, MIDAS, GAD-7, PHQ-9), multinational, n=5,954.",
     "Survey template Q1-Q3, Q12 use validated-screen style items and a disability count; REPORT.md warns to report screened-positive, never diagnosed, and to stratify by sex/faculty instead of convenience-only sampling."),
    ("Almomani et al. 2025", "Care gap in students: 82% used analgesics, only 11.7% sought medical help; 52.1% screened positive in a 94.2%-female sample.",
     "Survey template Q13-Q14 measure care-seeking and medicine use; REPORT.md uses Almomani for triggers/disability patterns, NOT for prevalence."),
    ("Angerhoefer et al. 2025", "Missed diagnosis is patterned: men diagnosed before referral 57.0% vs women 73.8%; pressing pain raises misses in men.",
     "Missed-diagnosis risk flags in Part C (below) and survey Q4-Q5, Q7 capture the atypical features (bilateral, pressing, no worsening with activity) that drive misses."),
    ("Suwanlaong et al. 2025", "43.3% misdiagnosed at first visit; first consult with a non-neurologist aOR 7.92; vertigo aOR 4.39.",
     "Missed-diagnosis risk flags in Part C; survey Q13 records who was consulted first. Wide CIs (male aOR 95% CI 1.07-56.50) are quoted, not hidden."),
    ("Elmazny et al. 2025", "Trigger awareness gap: stress recognised by 93.4%, but only 28.3% actively manage triggers and only 18% keep a diary.",
     "Survey template Q8-Q11 measure trigger awareness AND action (diary, sleep schedule, stress techniques) — knowledge alone is not counted as management."),
    ("Barbanti et al. 2025 (EMBRACE II)", "Real-world treatment evidence must show the denominator: >=50% responders 69%, but only for 74/215 (34%) who completed >=24 weeks.",
     "Part B prints the completers denominator next to every responder rate; REPORT.md never quotes 69% without '74/215'."),
    ("Cardona-Roca et al. 2025", "Patient satisfaction matters, not only migraine days: TSQM 77.2, MMD -8.7, 65% >=50% responders — in completers only.",
     "Part B compares Barbanti vs Cardona-Roca responder rates with their completers caveats side by side; survey Q14 asks about preventive use without recommending any drug."),
    ("Khan et al. 2024", "Cautionary lesson: augmenting 400 records to 1,447 and claiming DNN 99.66% — no external validation, likely leakage.",
     "NO augmentation anywhere in this project. Stratified split, 5-fold CV, per-class reporting, and this honesty note in the code, README and REPORT."),
]

# ------------------------------------------------------------------
# PART A — Real 400-patient classification
# ------------------------------------------------------------------
print("=" * 70)
print("PART A — Real 400-patient migraine classification")
print("=" * 70)
df = pd.read_csv("data/migraine.csv")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("\nMigraine type distribution:")
print(df["Type"].value_counts().to_string())
print("\nAge summary:")
print(df["Age"].describe().round(1).to_string())
print("\nSymptom rates (% of patients with symptom > 0):")
for col in ["Nausea", "Vomit", "Phonophobia", "Photophobia", "Visual", "Vertigo"]:
    print(f"  {col}: {(df[col] > 0).mean()*100:.1f}%")

# Charts — required
plt.figure(figsize=(9, 5))
df["Type"].value_counts().plot(kind="bar", color="#d14836")
plt.ylabel("Patients")
plt.title("Migraine type — 400 real physician-diagnosed patients")
plt.xticks(rotation=25, ha="right"); plt.tight_layout()
plt.savefig("outputs/type_distribution.png", dpi=150)

plt.figure(figsize=(8, 5))
symptoms = ["Nausea", "Vomit", "Phonophobia", "Photophobia", "Visual", "Vertigo", "Tinnitus"]
rates = [(df[c] > 0).mean()*100 for c in symptoms]
plt.barh(symptoms, rates, color="#2f6f4f")
plt.xlabel("Patients with symptom (%)")
plt.title("Most common migraine symptoms (400 real patients)")
plt.tight_layout()
plt.savefig("outputs/symptom_rates.png", dpi=150)

plt.figure(figsize=(8, 5))
df["Age"].plot(kind="hist", bins=15, color="#4a6fa5")
plt.xlabel("Age (years)"); plt.ylabel("Patients")
plt.title("Age distribution of migraine patients (mean 31.7, median 28)")
plt.tight_layout()
plt.savefig("outputs/age_distribution.png", dpi=150)

X = df.drop(columns=["Type"]); y = df["Type"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)} (stratified, random_state=42, NO augmentation)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000), True),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=42), False),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False),
}
model_rows = []
for name, (model, use_scaled) in models.items():
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s if use_scaled else X_test
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    acc = accuracy_score(y_test, pred)
    model_rows.append({"model": name, "test_accuracy_pct": round(acc*100, 1),
                       "cv_accuracy_pct": round(cv_scores.mean()*100, 1),
                       "cv_std_pct": round(cv_scores.std()*100, 1)})
    print(f"\n{name}:")
    print(f"  Test accuracy: {acc*100:.1f}%")
    print(f"  5-fold CV accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
    print(classification_report(y_test, pred, digits=2, zero_division=0))
    print("  Confusion matrix:\n", confusion_matrix(y_test, pred))

model_df = pd.DataFrame(model_rows)
model_df.to_csv("outputs/model_comparison.csv", index=False)

# ML model comparison chart
plt.figure(figsize=(8, 5))
x = np.arange(len(model_df))
plt.bar(x - 0.2, model_df["test_accuracy_pct"], 0.4, label="Test accuracy (%)", color="#d14836")
plt.bar(x + 0.2, model_df["cv_accuracy_pct"], 0.4, label="5-fold CV accuracy (%)", color="#4a6fa5")
plt.xticks(x, model_df["model"]); plt.ylabel("Accuracy (%)"); plt.ylim(0, 100)
plt.title("ML model comparison — honest, no augmentation (test n=80)")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)

# Feature importance
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 6))
imp.plot(kind="barh", color="#d14836")
plt.title("What matters most? — Random Forest feature importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
print("\nFeature importance (top 5):")
print(imp.sort_values(ascending=False).head(5).round(3).to_string())

# ------------------------------------------------------------------
# PART B — Evidence from the 10 papers
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART B — Evidence from the 10 papers (data/papers.csv)")
print("=" * 70)
papers = pd.read_csv("data/papers.csv")
print(papers[["rank", "short_cite", "year", "sample_n", "sample_unit"]].to_string(index=False))
print(f"\nTotal sample across the 10 papers (note: overlapping populations, do NOT add as one sample): {papers['sample_n'].sum():,}")
print(f"Largest sample: {papers.loc[papers['sample_n'].idxmax(), 'short_cite']} (n={papers['sample_n'].max():,})")
print(f"Smallest sample: {papers.loc[papers['sample_n'].idxmin(), 'short_cite']} (n={papers['sample_n'].min()})")
print("\nScreened prevalence (student surveys only):")
print(papers.dropna(subset=["prevalence_screened_pct"])[["short_cite", "prevalence_screened_pct"]].to_string(index=False))
print("\nML AUCs reported in the papers (forecasting, not classification):")
print(papers.dropna(subset=["ml_auc"])[["short_cite", "ml_auc", "ml_accuracy_pct"]].to_string(index=False))
print("\nTreatment >=50% responders (completers-only — see REPORT.md):")
print(papers.dropna(subset=["responders_ge50_pct"])[["short_cite", "responders_ge50_pct"]].to_string(index=False))
print(f"\nMisdiagnosed at first visit (Suwanlaong): {papers.loc[papers['short_cite']=='Suwanlaong et al.', 'misdiagnosed_pct'].iloc[0]}%")
print("Diagnosed before referral — men vs women (Angerhoefer): "
      f"{papers.loc[papers['short_cite']=='Angerhoefer et al.', 'diagnosed_before_men_pct'].iloc[0]}% vs "
      f"{papers.loc[papers['short_cite']=='Angerhoefer et al.', 'diagnosed_before_women_pct'].iloc[0]}%")

# Chart: paper sample sizes (log scale — Rabany n=53,065 dwarfs the rest)
plt.figure(figsize=(10, 5))
order = papers.sort_values("sample_n")
plt.barh(order["short_cite"], order["sample_n"], color="#4a6fa5")
plt.xscale("log")
plt.xlabel("Sample size (log scale)")
plt.title("Sample sizes across the 10 papers (log scale)")
plt.tight_layout()
plt.savefig("outputs/paper_sample_sizes.png", dpi=150)

# Chart: trigger rates across papers (sleep / noise where reported)
trig = papers.dropna(subset=["sleep_trigger_pct"]).copy()
plt.figure(figsize=(8, 5))
x = np.arange(len(trig))
plt.bar(x - 0.2, trig["sleep_trigger_pct"], 0.4, label="Sleep disturbance (%)", color="#d14836")
plt.bar(x + 0.2, trig["noise_trigger_pct"].fillna(0), 0.4, label="Noise (%) — 0 = not reported in that paper", color="#4a6fa5")
plt.xticks(x, trig["short_cite"]); plt.ylabel("Patients reporting trigger (%)")
plt.title("Trigger rates across papers — sleep is top everywhere it was asked\n(Elmazny noise 84.5% is awareness of noise as a trigger, not incidence)")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("outputs/trigger_rates.png", dpi=150)
print("\nTrigger rates (sleep):")
print(trig[["short_cite", "sleep_trigger_pct", "noise_trigger_pct"]].to_string(index=False))

# Chart: ML AUCs from the papers + this project's best test accuracy context
# (AUC and accuracy are different metrics — chart labels say so explicitly)
plt.figure(figsize=(8, 5))
auc_rows = [
    ("Rabany 2026\nXGBoost forecast", 0.893),
    ("Faisal 2026\ntime-series", 0.84),
    ("Faisal 2026\ndecision tree", 0.59),
    ("Faisal 2026\nfoundation model", 0.55),
]
labels, vals = zip(*auc_rows)
plt.barh(labels, vals, color="#2f6f4f")
plt.xlabel("AUC (0.5 = chance, 1.0 = perfect)")
plt.title("ML forecasting AUCs in the papers — time-series wins, plain models ~chance")
plt.xlim(0, 1); plt.tight_layout()
plt.savefig("outputs/paper_ml_aucs.png", dpi=150)

# ------------------------------------------------------------------
# PART C — Idea integration: missed-diagnosis risk flags (prototype)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART C — One idea from each paper (see outputs/idea_map.csv)")
print("=" * 70)
idea_df = pd.DataFrame(IDEA_MAP, columns=["paper", "idea_taken", "how_it_shaped_this_project"])
idea_df.to_csv("outputs/idea_map.csv", index=False)
for _, r in idea_df.iterrows():
    print(f"\n{r['paper']} — {r['idea_taken']}\n  -> {r['how_it_shaped_this_project']}")

# Prototype missed-diagnosis flags, from Angerhoefer + Suwanlaong predictors
# that ARE present in the 400-patient data. Documented as a prototype:
# - Vertigo present (Suwanlaong: vertigo/dizziness aOR 4.39)
# - No visual symptoms (atypical / no-aura presentation; Visual is the top ML feature)
# - Pressing-type character cannot be asserted from the coded Character column
#   without the codebook, so it is NOT used here — see REPORT.md limitations.
print("\nPrototype missed-diagnosis flags in the 400 patients (Angerhoefer/Suwanlaong predictors present in this data):")
n_vertigo = int((df["Vertigo"] > 0).sum())
n_no_visual = int((df["Visual"] == 0).sum())
n_both = int(((df["Vertigo"] > 0) & (df["Visual"] == 0)).sum())
print(f"  Vertigo present: {n_vertigo} patients ({n_vertigo/len(df)*100:.1f}%)")
print(f"  No visual symptoms: {n_no_visual} patients ({n_no_visual/len(df)*100:.1f}%)")
print(f"  Both (vertigo + no visual): {n_both} patients ({n_both/len(df)*100:.1f}%)")
print("  NOTE: sex, bilateral location, pressing character and first-consult clinician are NOT in this dataset,")
print("  so this flag is a prototype only — the survey template (Q4, Q5, Q7, Q13) collects them for future work.")
print("\nDone. Charts and tables saved in outputs/")
