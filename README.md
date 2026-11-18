# 🧠 Migraine — The BEST Project: Classification + Evidence + A Fixed Design

**Real patient data. Real code. Honest results. And a new project design that fixes what the 10 papers got wrong.**

This is not a copy of the papers. It takes **one idea from each of the 10 migraine papers** (2023–2026), finds **what was wrong in each paper**, and builds **one coherent project that fixes those mistakes in a new way**.

## 📊 The Real Data

### Part A — 400 real patients
- **Source:** Hospital Materno Infantil de Soledad, Colombia, 2013. Published via F1000Research / Code Ocean, Kaggle "Migraine Dataset".
- **Size:** 400 rows × 24 columns — 23 features + target `Type` (7 migraine types)
- **File:** `data/migraine.csv`
- **Types:** Typical aura with migraine 247, Migraine without aura 60, Familial hemiplegic 24, Typical aura without migraine 20, Basilar-type aura 18, Other 17, Sporadic hemiplegic 14
- **Age:** mean 31.7, median 28, range 15–77
- **Symptoms:** Nausea 98.8%, Photophobia 98.0%, Phonophobia 97.8%, Visual 79.5%, Vomit 32.2%, Vertigo 12.5%

### Part B — Evidence from the 10 papers
- **File:** `data/papers.csv` — sample sizes, prevalence, trigger rates, AUCs, responder rates, misdiagnosis rates, copied exactly from the research report, with a source URL for every paper.
- **Largest sample:** Rabany et al. 2026, n=53,065 app users. **Smallest:** Cardona-Roca et al. 2025, n=80.
- **Do NOT add the sample sizes together** — the populations overlap and are different kinds of studies. The code prints the arithmetic sum (62,871) only with that warning; this README does not use it as a finding.

## 🤖 Machine Learning Results — Honest, No Augmentation

Test set: 80 patients (20%, stratified, random_state=42). No data augmentation.

| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|
| **Logistic Regression** | **93.8%** | **92.2% (±4.1%)** |
| Gradient Boosting | 92.5% | 90.3% (±3.8%) |
| Random Forest | 88.8% | 89.7% (±3.4%) |

- **Best model:** Logistic Regression — simple, interpretable, best on test and CV.
- **Most important feature:** Visual symptoms (Random Forest importance 0.248), then Intensity (0.102), Age (0.085).
- **Weak spot, reported honestly:** rare types are hard — Familial hemiplegic recall 60% (5 test patients), Basilar-type recall 75% (4 test patients). Per-class results are in `outputs/results.txt`.

> ⚠️ **Khan et al. (2024) reported 99.66% on this same 400-record dataset after augmenting it to 1,447 records. That figure is NOT claimed here.** Synthetic near-duplicates can leak across train/test splits and inflate accuracy, there was no external validation, and the paper is internally inconsistent (Random Forest 98.50% in the abstract vs 88.50% in Table 3).

## 💡 One Idea From Each Paper — And How This Project Fixes Its Mistake

Full detail is in `REPORT.md` §3 — one subsection per paper: **Idea taken → What was wrong → How this project fixes it**. In short:

| Paper | Idea taken | What was wrong | Fix in this project |
|---|---|---|---|
| Rabany 2026 | Forecasting; 30-day history (37.64%) beats prodrome (11.13%) | Nerivio-only cohort; self-reported outcome incl. mild pain; no external validation; Theranica-affiliated authors | No fake time-series from cross-sectional data; a 4–8 week daily-diary design (Q15) so a future forecaster has real history |
| Faisal 2026 | How you model time matters: time-series AUC 0.84 vs tree 0.59 | Trial-selected episodic-only volunteers; 3 months; research sEMG sensor, not a smartwatch | Head-to-head model comparison on the same split/CV; daily-diary data proposed, not assumed |
| Samir 2025 | Student survey with validated screens (MS-Q, MIDAS, GAD-7, PHQ-9), n=5,954 | Convenience/snowball sampling; screening, not diagnosis; 72.5% "severe depression" likely a screening-threshold artefact | Stratified sampling by sex/faculty in the diary design; report "screened-positive", never "diagnosed" |
| Almomani 2025 | Care gap: 82% analgesics, only 11.7% sought care | 94.2%-female convenience sample; 52.1% "prevalence" vs ~16.1% meta-analytic | Used for triggers/disability only, never for prevalence; survey Q13–Q14 measure care-seeking |
| Angerhöfer 2025 | Missed diagnosis is patterned: men 57.0% vs women 73.8% diagnosed before referral | Single tertiary centre; retrospective; biological sex only | Prototype missed-diagnosis flags + survey Q4, Q5, Q7 capture bilateral/pressing/no-worsening features |
| Suwanlaong 2025 | 43.3% misdiagnosed; non-neurologist first consult aOR 7.92 | n=90, single centre; male aOR CI 1.07–56.50 is essentially unquantified | Flags + Q13 record first clinician; wide CIs quoted, not hidden |
| Elmazny 2025 | Trigger awareness ≠ action: 93.4% know stress, only 28.3% manage triggers, 18% keep a diary | Support-group selection bias; self-report; Egypt only | Survey Q8–Q11 measure awareness AND action separately |
| Barbanti 2025 | Real-world treatment: MMD −10.5, 69% ≥50% responders | Completers-only: 69% is for 74/215 (34%); no control group | Responder rate never quoted without "74/215" |
| Cardona-Roca 2025 | Satisfaction matters: TSQM 77.2, MMD −8.7, 65% responders | Completers ≥1 year only; 88.8% one drug; "100% adherence" is a refill proxy | Compared side-by-side with Barbanti, with both completers caveats |
| Khan 2024 | Cautionary lesson | Augmentation 400→1,447, claimed 99.66%, no external validation | **No augmentation at all**; stratified split + 5-fold CV + per-class reporting |

## 📁 Files

- `analysis.py` — full real code: Part A classification, Part B papers analysis, Part C idea map + missed-diagnosis flags
- `data/migraine.csv` — real dataset (400 patients)
- `data/papers.csv` — published numbers from the 10 papers, with URLs
- `data/survey_template.csv` — 15-question student survey built from the papers' ideas (Q1–Q15)
- `outputs/` — 8 charts, `model_comparison.csv`, `idea_map.csv`, `results.txt`
- `REPORT.md` — big full report: abstract, literature review, per-paper fix section, new design, results, discussion, limitations, references, viva Q&A
- `requirements.txt` — pandas, numpy, matplotlib, scikit-learn

## ▶️ How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

## 👤 Author

**Md Emon Rana** — BSc in CSE, Southeast University, Dhaka, Bangladesh
Data Science • AI • Machine Learning • Python

> ⚠️ This is a learning / decision-support project, not a medical diagnosis tool. A real doctor must always confirm a diagnosis.
