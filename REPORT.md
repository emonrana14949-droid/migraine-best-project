# Migraine Classification and Evidence Synthesis: A Fixed, Real-Life Project

**Md Emon Rana — BSc in Computer Science and Engineering, Southeast University, Dhaka, Bangladesh**

---

## Abstract

Migraine is one of the most common neurological disorders in the world, yet it is frequently misdiagnosed, under-treated, and studied with methods that over-claim. This project does three things in one coherent design. First, it classifies migraine type from symptoms using **400 real physician-diagnosed patient records** (Hospital Materno Infantil de Soledad, Colombia, 2013), with three machine learning models evaluated honestly — stratified split, 5-fold cross-validation, per-class reporting, and **no data augmentation**. Logistic Regression was best: **93.8% test accuracy and 92.2% 5-fold cross-validation accuracy**, ahead of Gradient Boosting (92.5% / 90.3%) and Random Forest (88.8% / 89.7%). Visual symptoms were the most important feature (Random Forest importance 0.248). Second, it analyses the published numbers of **10 recent migraine papers (2023–2026)** — sample sizes, screened prevalence, trigger rates, forecasting AUCs, treatment responder rates and misdiagnosis rates — in `data/papers.csv`, with a source URL for every paper. Third, and most importantly, it takes **one idea from each of the 10 papers**, states **what was wrong in each paper**, and shows **exactly how this new project fixes it** (Section 3). The fixes include: no augmentation and no leakage; stratified sampling instead of convenience-only sampling in the proposed student diary design; a clinical-confirmation subsample instead of screening-only prevalence; within-person case-crossover trigger analysis instead of cross-sectional correlation; per-class reporting for rare migraine types; and responder rates always quoted with their completers denominator. A 15-question student survey template (`data/survey_template.csv`) and a 4–8 week prospective diary design turn those fixes into a study a student team can actually run. The widely quoted 99.66% accuracy of Khan et al. (2024) on this same 400-record dataset is **not** claimed here, for reasons documented in Sections 2.10 and 3.10.

**Keywords:** migraine, classification, machine learning, logistic regression, forecasting, student survey, misdiagnosis, triggers, real-life data, honesty in ML

---

## 1. Introduction

### 1.1 What is migraine?

Migraine is a recurrent headache disorder, usually with moderate to severe throbbing pain, often on one side of the head, with nausea, vomiting, and sensitivity to light (photophobia) and sound (phonophobia). Some patients experience aura — visual or sensory disturbances before the headache. The International Headache Society (ICHD-3) classifies migraine into types, including migraine with aura, migraine without aura, familial hemiplegic migraine, sporadic hemiplegic migraine, and basilar-type aura.

### 1.2 Why does this matter in real life?

- Migraine affects about 1 in 7 people worldwide and is a leading cause of disability in young adults.
- Students are heavily affected: in a 2025 MENA study of 5,954 university students, 26.1% screened positive for migraine, and 23.2% of them had severe disability (Samir et al. 2025). In Jordan, only 11.7% of affected students sought medical help (Almomani et al. 2025).
- Diagnosis is often missed: 43.3% were misdiagnosed at first visit in Thailand (Suwanlaong et al. 2025), and men were diagnosed before tertiary referral in only 57.0% of cases vs 73.8% for women in Berlin (Angerhöfer et al. 2025).
- Sleep disturbance (59–83%) and stress are the top-reported triggers across independent studies, yet only 18% of patients keep a headache diary and only 28.3% actively manage triggers (Elmazny et al. 2025).

### 1.3 Problem statement

Three problems run through the recent migraine literature: (1) classification models over-claim accuracy by augmenting tiny datasets; (2) surveys quote screening-positive rates as if they were diagnosed prevalence, from convenience samples; (3) trigger studies report cross-sectional correlations as if they were causes, while diagnosis studies ignore the patients who are missed. Can one student project — built from real patient data and the published evidence — classify migraine type honestly, compare the literature fairly, and design the *next* study so it does not repeat those mistakes?

### 1.4 Objectives

1. Classify migraine type from symptoms on 400 real patients, honestly evaluated.
2. Analyse and compare the published numbers of 10 recent migraine papers.
3. Take one idea from each paper and state, per paper, what was wrong and how this project fixes it.
4. Produce a fixed, runnable student study design (diary cohort + survey template) that implements those fixes.

---

## 2. Literature Review — The 10 Papers

All numbers below are copied from the research report (`~/workspace/research_notes/migraine-research-papers-20261118-1348/report.md`, read 18 November 2026). Full URLs are in Section 11.

### 2.1 Rabany et al. (2026) — Next-day migraine forecasting from app diaries

*Neurology Open Access* 2(3):e000144. Retrospective ML analysis of prospective electronic diaries in the Nerivio app, Jan 2020–Jul 2025. **n=53,065 users, 770,473 report days**; 60/20/20 user-level train/validation/test split; 7 classifiers compared. Optimised XGBoost: precision 0.912, accuracy 0.810, sensitivity 0.800, specificity 0.830, **AUC 0.893**, F1 0.852. Rolling mean headache severity over 30 days contributed 37.64% of performance; 30-day features overall 56.34%; prodromal symptoms only 11.13%; current-day information 4.38%. No significant performance difference by sex or adult vs adolescent.

### 2.2 Samir et al. (2025) — Migraine burden in MENA university students

*BMC Public Health* 25:3605. Multinational cross-sectional online survey, Aug–Dec 2024, 11 low/middle-income MENA countries; convenience + snowball sampling; validated tools (MS-Q screening, MIDAS, GAD-7, PHQ-9). **n=5,954 students.** 26.1% screened positive (Iraq 38.9%, Algeria 31.5%, Egypt 19.9%, Morocco 18.4%). Triggers: sleep disturbance 59.7%, noise 47.4%, sun exposure 45.6%. Among positives: 23.2% severe disability, 29.9% moderate anxiety, 72.5% severe depression. Predictors: female, older age, non-medical field.

### 2.3 Faisal et al. (2026) — Time-series ML forecasting from mHealth + wearables

*Journal of Headache and Pain* 27:91. Secondary ML analysis of the BioCer randomised trial (Norway, Jan 2023–Sep 2024). **n=146 individuals, 21,550 headache days**; daily diary + wearable sensors (trapezius sEMG, heart-rate variability, skin temperature). Best time-series model **AUC 0.84** (95% CI 0.82–0.85) next day, 0.76 for a 3-day window — vs decision tree 0.59 and foundation model 0.55. Most important features: headache intensity, headache duration, heart-rate scores.

### 2.4 Angerhöfer et al. (2025) — Sex disparities in migraine recognition

*Journal of Headache and Pain* 26:225. Retrospective cross-sectional analysis, Charité Headache Center, Berlin, Dec 2015–Jan 2023. **n=1,130 (82% women, 18% men; median age 42.3)**, clinically diagnosed. Men were diagnosed before referral in **57.0% vs 73.8%** for women (OR 0.47, p<0.001). Men reported lower pain intensity and shorter attacks; women more often had unilateral headache, nausea/vomiting, photophobia and osmophobia. In men, pressing (non-pulsating) headache quality substantially raised the odds of being missed.

### 2.5 Barbanti et al. (2025) — EMBRACE II, real-world eptinezumab

*Journal of Neurology* 272:382. Prospective, multicentre (22 centres), 24-week real-world study; patients who had failed >3 preventives; eptinezumab 100 mg quarterly. **n=215 received ≥1 dose; only n=74 treated ≥24 weeks and included in the effectiveness analysis.** Monthly migraine/headache days −10.5, analgesic intake −15.6, HIT-6 −9.9, MIDAS −48.7 (all p<0.001). **≥50% responders 69%**, ≥75% 39.2%, 100% 4.1%. Adverse events infrequent (2.8%).

### 2.6 Elmazny et al. (2025) — Migraine triggers, lifestyle, and provider education

*Journal of Headache and Pain* 26:189. Cross-sectional online survey (Google Forms, Arabic), Feb–Apr 2025, Egyptian migraine support groups. **n=515, median age 31, 81.9% female**, formal diagnosis >1 year. Only **43%** felt they had sufficient knowledge of triggers. Awareness: stress 93.4%, under-sleeping 85.6%, irregular sleep 86.0%, loud noises 84.5%; only ~40% recognised food triggers. 63.1% said physicians had discussed lifestyle changes, yet only **28.3% actively managed triggers**; consistent sleep schedule 54.0%, stress techniques 49.5%, specific diet 34.0%; only **18% keep a migraine diary**.

### 2.7 Almomani et al. (2025) — Migraine and disability in Jordanian university students

*Journal of Pain Research* 18:2093–2104. Cross-sectional online survey via social media, Aug 2022–Feb 2023; convenience sample. **n=1,276 (94.2% female, median age 21, 94.8% undergraduates).** 65.0% reported headache in the past 3 months; 80.2% of those screened positive (52.1% of the whole sample), 46% with aura. Photophobia ~90% (commonest symptom); sleep disturbance the top trigger (83%). 82% used analgesics, mainly OTC (66.8%); only **11.7% sought medical help**; 40.9% had increased analgesic dose over time; only 12.2% used prophylaxis. 27.5% had severe (MIDAS IV) disability.

### 2.8 Cardona-Roca et al. (2025) — Anti-CGRP satisfaction and quality of life

*BMC Neurology* 25:418. Single-centre retrospective study, Hospital General de Granollers, Spain; patients treated ≥1 year with the same anti-CGRP mAb. **n=80 (95% women, mean age 50 ± 9.9; 70% chronic migraine; 88.8% fremanezumab).** Mean global treatment satisfaction (TSQM) **77.2 ± 20.8**; monthly migraine days −8.7 ± 7.4; **65% achieved ≥50% reduction**; 66% achieved ≥6-point HIT-6 reduction (mean −12.1); 22.5% reported mild adverse events; reported adherence 100%.

### 2.9 Suwanlaong et al. (2025) — Why migraine is misdiagnosed

*Journal of Headache and Pain* 27:35 (2026 issue). Prospective cross-sectional study, Songkhla Hospital, Thailand, Jul 2024–Apr 2025; final diagnosis confirmed by two blinded independent neurologists (ICHD-3). **n=90 (87.8% female, mean age 43.4 ± 14.8).** **43.3% had been misdiagnosed or undiagnosed at first visit**; commonest wrong diagnoses tension-type headache and sinusitis. Predictors of inaccurate diagnosis: male sex aOR 7.77 (95% CI 1.07–56.50), bilateral headache aOR 3.90 (1.25–12.13), no worsening by physical activity aOR 5.09 (1.54–16.89), vertigo/dizziness aOR 4.39 (1.22–15.83), first consultation with a non-neurologist aOR 7.92 (2.63–23.88).

### 2.10 Khan et al. (2024) — ML classification with data augmentation (cite with caution)

*Scientific Reports* 14:5180. ML classification on the public Kaggle dataset of symptom records — the same 400 records used in Part A of this project. **n=400 original records → 1,447 after augmentation; test set appears to be 80 instances.** Reported with augmentation: DNN 99.66%, KNN 97.10%, SVM 94.60%, decision tree 88.20%, Random Forest 98.50% (abstract) — but the paper's own Table 3 lists Random Forest as 88.50%, an internal inconsistency. No explicit limitations section, no external validation, no prospective or clinical test, no calibration analysis, and the dataset's labels/provenance are not clinically verified in the paper.

---

## 3. What Was Wrong in Each Paper — And How This New Project Fixes It (The New Way)

This is the section that makes this project new. For each paper: **(a) the idea taken, (b) the limitation / what was wrong, (c) exactly how this project fixes it.** A summary table follows in §3.11.

### 3.1 Rabany et al. (2026)

- **Idea taken:** Forecasting, not just classifying. Long-term headache history matters: 30-day severity contributed 37.64% of model performance, 30-day features overall 56.34%, while prodromal symptoms contributed only 11.13%.
- **What was wrong:** Single-device (Nerivio) user cohort — not the general migraine population. Outcome is self-reported and includes mild pain (the sensitivity analysis without mild pain was weaker). Sparse reporting (mean 5.19 reports per 30 days). No trigger, sleep, stress, menstrual or diet features. No external validation. No test of whether a forecast changes behaviour or outcomes. Many authors are affiliated with Theranica, maker of Nerivio — a conflict of interest to declare when citing.
- **How this project fixes it:** This project does **not** fake a time-series from the 400 cross-sectional records — there is no daily history in that dataset, and inventing one would be exactly the kind of over-claim Rabany warns against by example. Instead, the fix is built into the *design*: the proposed diary cohort (§4.3, survey Q15) collects 4–8 weeks of daily data (sleep, stress, headache, medicine, classes missed) so a future forecasting model has the real 30-day history it needs, and the ML evaluation in Part A reports AUC-style honesty (test accuracy *and* 5-fold CV, per-class recall) instead of a single headline number. When this project's forecasting idea is used, it must be tested on whether the forecast changes disability or medicine use — the endpoint Rabany never tested.

### 3.2 Faisal et al. (2026)

- **Idea taken:** How you model time matters more than adding more algorithms — time-series AUC 0.84 vs decision tree 0.59 and foundation model 0.55 on the same data.
- **What was wrong:** Secondary analysis of a trial cohort — high-adherence, motivated, episodic-migraine-only volunteers. Only ~3 months of data. The biofeedback treatment itself may alter the signals being predicted. Trapezius sEMG is a research sensor, not a consumer smartwatch, so results do not transfer directly to Apple Watch-type data. No external validation cohort. Sensitivity remained limited — a good AUC does not mean most attacks are caught.
- **How this project fixes it:** Part A copies Faisal's *method*, not his numbers: three models are compared **head-to-head on the same stratified split and the same 5-fold CV** (Logistic Regression 93.8% / 92.2% CV, Gradient Boosting 92.5% / 90.3%, Random Forest 88.8% / 89.7%), so the winner is the winner on identical ground — and the simplest model wins, which is reported instead of hidden. The diary design collects daily data with ordinary phones (Q15), not a research sensor, so a future time-series model is tested on data a student can actually collect. Sensitivity/recall is reported per class (Section 6.3), so a good headline accuracy cannot hide missed cases.

### 3.3 Samir et al. (2025)

- **Idea taken:** Student survey design done at scale — validated screens (MS-Q, MIDAS, GAD-7, PHQ-9), multinational (11 countries), n=5,954.
- **What was wrong:** Convenience + snowball sampling — selection bias and uneven country representation. Screening (MS-Q) without clinical confirmation, so "26.1%" is screened-positive, not diagnosed prevalence, and misclassification is likely. Cross-sectional, so no causal claims about caffeine, sleep or study hours are valid. Pilot Cronbach's alpha only 0.72 (MS-Q) / 0.71 (MIDAS). Many statistical tests raise Type I error risk. The 72.5% "severe depression" figure is implausibly high for a student sample and likely reflects screening-tool thresholds, not clinical depression.
- **How this project fixes it:** The diary design (§4.3) uses **stratified sampling by sex and faculty, not convenience-only social media**; it screens with ID-Migraine/MS-Q style items (survey Q1–Q3) **and confirms a subsample against ICHD-3 criteria** with a clinician or validated structured interview; it reports **"screened-positive", never "diagnosed"**; it pre-registers its hypotheses and limits the number of tests; and it treats comorbidity scores (PHQ-9/GAD-7) as screening scores, not clinical diagnoses. In this project's own analysis, Samir's 26.1% is quoted as screened-positive with the caveat attached, every time.

### 3.4 Almomani et al. (2025)

- **Idea taken:** The care gap in students — 82% used analgesics, only 11.7% sought medical help; triggers and disability patterns in a student population.
- **What was wrong:** Convenience online sampling with recall bias. No clinical confirmation of diagnosis. The sample is **94.2% female**, so the 52.1% "prevalence" is not a population estimate — the meta-analytic student prevalence is ~16.1% — and sex comparisons are impossible. Data are from 2022–23, published 2025.
- **How this project fixes it:** Almomani is used in this project **for triggers and disability patterns only, never for prevalence** — the 52.1% figure is always paired with "94.2%-female convenience sample" in this report. The diary design stratifies by sex so a 94%-female sample cannot happen by accident, and survey Q13–Q14 directly measure the care gap (who was consulted, what medicine was taken, dose escalation) instead of assuming it. If the diary study ever reports a prevalence, it will be reported with its sampling frame, response rate and sex/faculty breakdown — the three things Almomani's headline number lacks.

### 3.5 Angerhöfer et al. (2025)

- **Idea taken:** Missed diagnosis is patterned, not random — men diagnosed before referral 57.0% vs women 73.8%; pressing (non-pulsating) headache raises the odds of being missed in men.
- **What was wrong:** Single tertiary centre in Germany — referral bias toward severe, treatment-resistant cases, so results may not apply to primary care. Retrospective, cross-sectional. Analysis by biological sex only, no gender-identity data. No stratification by age/hormonal life stage, which may confound presentation.
- **How this project fixes it:** The pattern is turned into **prototype missed-diagnosis flags in code** (Part C of `analysis.py`): in the 400 patients, vertigo present in 50 patients (12.5%), no visual symptoms in 82 patients (20.5%), both in 5 patients (1.2%). The flags are labelled *prototype* because sex, bilateral location, pressing character and first-consult clinician are **not** in the 400-patient dataset — and the code says so in plain text instead of pretending otherwise. The survey template collects exactly those missing fields (Q4 bilateral vs one-sided, Q5 pulsating vs pressing, Q7 worsening with activity, Q13 first clinician), so the *next* study can test Angerhöfer's pattern in primary-care-age students instead of a tertiary centre. The diary design also records age band, so presentation can be stratified by age.

### 3.6 Suwanlaong et al. (2025)

- **Idea taken:** 43.3% misdiagnosed at first visit; the strongest predictor was first consultation with a non-neurologist (aOR 7.92), plus vertigo (aOR 4.39) and bilateral headache (aOR 3.90).
- **What was wrong:** n=90 from a single tertiary hospital — small and referral-biased. The headline male-sex effect rests on ~11 men and a confidence interval spanning **1.07–56.50, i.e. essentially unquantified**. Prospective enrolment but cross-sectional analysis of past diagnostic history (recall bias about the first visit). Thai tertiary setting limits generalisability to primary care elsewhere.
- **How this project fixes it:** The 43.3% figure is quoted with n=90 and the single-centre caveat, and the male aOR 7.77 is **always quoted with its full CI 1.07–56.50**, never as a bare "7.77 times more likely". The diary design is powered for more than 90 students (150–300 proposed) and records first-consult clinician prospectively (Q13, asked at enrolment, not recalled months later), so the non-neurologist predictor can be tested without recall bias. The missed-diagnosis flags (Part C) use vertigo — a predictor present in the 400-patient data — and openly list the predictors that are *not* present, instead of silently dropping them.

### 3.7 Elmazny et al. (2025)

- **Idea taken:** Trigger awareness is not trigger management — stress recognised by 93.4%, but only 28.3% actively manage triggers and only 18% keep a diary.
- **What was wrong:** Selection bias — support-group members are more engaged than average patients. Self-report with no verification of diagnosis or behaviour. Online format excludes people without internet/digital literacy. Cross-sectional, so knowledge cannot be linked to attack frequency. Egyptian sample only.
- **How this project fixes it:** Survey Q8–Q11 measure **awareness and action separately**: Q8–Q10 ask how often sleep/stress/noise came before a headache, Q11 asks whether the student keeps a diary, and the diary design measures whether a 1-page trigger-education handout + diary *changes* trigger-management behaviour at 4 weeks (before/after or randomised handout vs no-handout pilot). Knowledge alone is never counted as management in this project. In Part B, Elmazny's noise figure (84.5%) is labelled **awareness**, not incidence — the chart title in `analysis.py` says so explicitly — because mixing awareness with incidence is exactly how trigger numbers get misquoted.

### 3.8 Barbanti et al. (2025) — EMBRACE II

- **Idea taken:** Real-world treatment evidence in refractory patients — monthly migraine days −10.5, ≥50% responders 69%.
- **What was wrong:** No control or placebo group, open-label — placebo and regression-to-the-mean cannot be excluded. Effectiveness is reported only for the **74/215 (34%)** who completed ≥24 weeks, so the 69% responder rate is a completers-only figure and overstates benefit in all starters. 24 weeks is short for a preventive. Highly selected refractory population, not general migraine.
- **How this project fixes it:** In this project, **69% is never quoted without "74/215"**. Part B prints the completers denominator next to every responder rate, the REPORT and README repeat it, and the diary design includes an **attrition flow** (enrolled → completed → analysed) as a required output — the EMBRACE II warning turned into a rule. The project also makes no treatment recommendation: survey Q14 asks about preventive use without recommending any drug, because an uncontrolled, completers-only study cannot support a recommendation.

### 3.9 Cardona-Roca et al. (2025)

- **Idea taken:** Patient satisfaction matters, not only migraine days — TSQM 77.2, monthly migraine days −8.7, 65% ≥50% responders.
- **What was wrong:** Retrospective, single centre, no control group. Only patients who *stayed* on the same drug for a year are included (survivor/completer bias — non-responders who stopped are invisible). 88.8% on one drug (fremanezumab), so no comparison between antibodies is possible. 95% women, mean age 50 — not students/young adults. "100% adherence" by Medication Possession Ratio is a pharmacy-refill proxy, not observed intake, and is implausibly perfect for routine care.
- **How this project fixes it:** Barbanti (69%, 74/215) and Cardona-Roca (65%, completers ≥1 year) are **compared side-by-side in Part B with both completers caveats attached**, instead of being quoted as two independent "65–69%" successes. The "100% adherence" figure is quoted with its refill-proxy explanation, never as observed adherence. The diary design measures satisfaction and burden (MIDAS, HIT-6) alongside headache days, following Cardona-Roca's idea — but in students, the population Cardona-Roca does not cover, and with dropouts counted, not hidden.

### 3.10 Khan et al. (2024)

- **Idea taken:** None as evidence — taken as a **cautionary lesson**: this is the paper that shows what not to do.
- **What was wrong:** A 400-record public dataset, synthetically inflated to 1,447, is far too small for a 99.66% claim — synthetic/augmented near-duplicates across train/test splits can leak and inflate accuracy. No external validation, no prospective or clinical test, no calibration or confidence analysis, and the dataset's labels/provenance are not clinically verified. No explicit limitations section. Internal inconsistency: Random Forest 98.50% in the abstract vs 88.50% in Table 3.
- **How this project fixes it:** **No augmentation anywhere.** Stratified 80/20 split (320 train / 80 test, random_state=42), 5-fold cross-validation, per-class precision/recall/F1 with support counts, confusion matrices for all three models, and the honest result: Logistic Regression 93.8% test / 92.2% CV, Gradient Boosting 92.5% / 90.3%, Random Forest 88.8% / 89.7%. The 99.66% figure appears in this project only in the honesty warning, never as a result. The diary design pre-registers that a 99% accuracy on a tiny set would signal leakage, not success.

### 3.11 Summary — the fixed version at a glance

| # | Paper | Limitation (one line) | Fix in this project (one line) |
|---|---|---|---|
| 1 | Rabany 2026 | Nerivio-only; self-report; no external validation | No fake time-series; real 4–8 week diary proposed for forecasting |
| 2 | Faisal 2026 | Trial-only volunteers; research sensor; 3 months | Head-to-head comparison on same split/CV; daily data from ordinary phones |
| 3 | Samir 2025 | Convenience/snowball; screening ≠ diagnosis | Stratified sampling; clinical-confirmation subsample; "screened-positive" wording |
| 4 | Almomani 2025 | 94.2% female; 52.1% is not prevalence | Used for triggers/disability only; sex-stratified design |
| 5 | Angerhöfer 2025 | Single tertiary centre; retrospective | Prototype flags + survey items for atypical features; age stratification |
| 6 | Suwanlaong 2025 | n=90; male aOR CI 1.07–56.50 | CIs quoted in full; prospective first-clinician record; n=150–300 proposed |
| 7 | Elmazny 2025 | Support-group bias; awareness ≠ action | Awareness and action measured separately; handout + diary pilot |
| 8 | Barbanti 2025 | 69% is 74/215 completers-only; no control | Denominator always quoted; attrition flow required |
| 9 | Cardona-Roca 2025 | Completers-only; "100% adherence" is a refill proxy | Side-by-side comparison with caveats; dropouts counted |
| 10 | Khan 2024 | Augmentation 400→1,447; claimed 99.66% | No augmentation; stratified split + CV + per-class reporting |

**The new way, in one sentence:** this project is the fixed version of those 10 papers — same questions, same real data, but with the sampling, confirmation, time-series, attrition and honesty failures removed by design, not by apology afterwards.

---

## 4. The New Project Design (The Fixed Version)

### 4.1 Design in one paragraph

One coherent project with three parts that reinforce each other: **(A)** honest ML classification on 400 real patients; **(B)** a quantitative synthesis of the 10 papers' published numbers, with every caveat attached; **(C)** a fixed student study design — a 4–8 week prospective diary cohort (150–300 students, stratified by sex and faculty) with a 15-question survey — that collects the data the 10 papers were missing. Part A proves the team can do honest ML. Part B proves the team understands the literature, including its flaws. Part C proves the team knows what to do next.

### 4.2 Part A — Classification (done, code in `analysis.py`)

400 patients, 7 migraine types, 3 models, stratified split, 5-fold CV, no augmentation. Results in Section 6.

### 4.3 Part C — Proposed diary cohort (the fixed study)

- **Recruit:** 150–300 students, **stratified by sex and faculty**, not convenience-only social media (fixes Samir/Almomani).
- **Screen:** ID-Migraine / MS-Q style items (survey Q1–Q3); **confirm a subsample against ICHD-3** with a clinician or validated structured interview (fixes Samir/Almomani).
- **Collect daily for 4–8 weeks (2-minute diary):** sleep hours/quality, stress 0–10, caffeine, meals/skipped meals, screen time, weather, headache presence/intensity, photophobia/phonophobia/nausea, medication taken, classes missed (fixes Rabany/Faisal — real history for forecasting; fixes Elmazny — diary use measured, not assumed).
- **Analysis 1 — triggers, properly:** within-person **case-crossover** analysis — compare trigger exposure on days before an attack vs the same student's attack-free days. This avoids the cross-sectional confounding that weakens Samir/Almomani/Elmazny and can separate prodrome from trigger (fixes §3.3, §3.4, §3.7).
- **Analysis 2 — ML, honestly:** replicate Faisal/Rabany on the diary data — simple baseline (logistic regression) vs time-series/gradient-boosting model with **user-level** train/test split; report AUC *and* sensitivity/specificity; pre-register that 99% accuracy on a tiny set signals leakage (fixes Khan, §3.10).
- **Analysis 3 — burden + care gap:** MIDAS + HIT-6 + PHQ-9/GAD-7 (as in Samir) plus "did you seek care / what did you take" (as in Almomani). Test whether a 1-page trigger-education handout + diary changes trigger-management behaviour at 4 weeks — a small before/after or randomised handout vs no-handout pilot (fixes Elmazny; first student intervention data in this set).
- **Missed-diagnosis check:** survey Q4, Q5, Q7, Q13 capture bilateral pain, pressing quality, no worsening with activity, and first clinician — the Angerhöfer/Suwanlaong predictors — and report them with full confidence intervals (fixes §3.5, §3.6).
- **Attrition flow, always:** enrolled → completed → analysed, with reasons for dropout (fixes Barbanti/Cardona-Roca, §3.8, §3.9). No responder rate without its denominator.
- **What to avoid (checklist):** no convenience-only sampling quoted as prevalence; no 94%-female sample; no augmentation claiming 99%; no calling screening-positive "diagnosed"; no hiding wide CIs.

---

## 5. Dataset

### 5.1 Part A data — 400 real patients

- **Origin:** 400 medical records of patients diagnosed with migraine, recorded by trained medical personnel at Hospital Materno Infantil de Soledad, Colombia, first quarter of 2013 (De la Hoz et al. 2014 master's thesis).
- **Publication:** F1000Research (Sanchez-Sanchez et al. 2020), Code Ocean capsule DOI 10.24433/CO.2826453.v1, Kaggle "Migraine Dataset" (ranzeet013/migraine-dataset), Frontiers Data Sheet 1 (the file used here, `data/migraine.csv`).
- **Size:** 400 rows × 24 columns. 23 features + 1 target. No patient-identifiable data.
- **Target — migraine type (7 classes):** Typical aura with migraine 247 (61.8%), Migraine without aura 60 (15.0%), Familial hemiplegic migraine 24 (6.0%), Typical aura without migraine 20 (5.0%), Basilar-type aura 18 (4.5%), Other 17 (4.3%), Sporadic hemiplegic migraine 14 (3.5%). **Class imbalance is the central challenge:** a model that always predicted the majority class would already score 61.8% accuracy — so accuracy alone is misleading and per-class recall must be reported.

### 5.2 Part B data — the 10 papers

`data/papers.csv` — 10 rows, one per paper, with rank, citation, year, journal, sample_n, sample_unit, study_design, screened prevalence, sleep/noise trigger rates, ML AUC, ML accuracy, ≥50% responder rate, misdiagnosed rate, diagnosed-before-referral by sex, key finding, and source URL. Numbers were copied exactly from the research report; blank means "not reported in that paper", not zero. The arithmetic sum of sample_n is 62,871 — **that sum is not a finding**, because the populations overlap and are different kinds of studies, and it is never used as one in this report.

### 5.3 Survey template

`data/survey_template.csv` — 15 questions (Q1–Q15) in 6 sections (Screening, Symptoms, Triggers, Burden, Care-seeking, Diary follow-up), each question tagged with the paper idea it came from. See §4.3 and the file itself.

---

## 6. Methodology

### 6.1 Tools

Python, Pandas, NumPy, Matplotlib, scikit-learn. Full code in `analysis.py`, reproducible with `random_state=42`. Run with the project venv: `~/workspace/medical-diabetes-real-life/venv/bin/python analysis.py`.

### 6.2 Part A — exploratory analysis

Type distribution, age distribution, and symptom rates were computed and charted (`outputs/type_distribution.png`, `outputs/age_distribution.png`, `outputs/symptom_rates.png`).

### 6.3 Part A — ML evaluation (honest)

- Split: 80% train (320 patients), 20% test (80 patients), `stratify=y`, `random_state=42`.
- **No data augmentation.**
- Models: (1) Logistic Regression — features standardised with `StandardScaler`, `max_iter=2000`; (2) Random Forest — 300 trees; (3) Gradient Boosting — default scikit-learn parameters.
- Evaluation: test accuracy, per-class precision/recall/F1 with support counts, confusion matrices, and **5-fold stratified cross-validation** on the training set.

### 6.4 Part B — papers analysis

`data/papers.csv` was loaded in `analysis.py` Part B: sample sizes (log-scale chart, because Rabany n=53,065 dwarfs the rest), screened prevalence (student surveys only), trigger rates, ML AUCs, treatment responder rates with completers denominators, and the diagnosis gap (men vs women diagnosed before referral; misdiagnosed rate). Charts: `outputs/paper_sample_sizes.png`, `outputs/trigger_rates.png`, `outputs/paper_ml_aucs.png`.

### 6.5 Part C — idea map and prototype flags

`outputs/idea_map.csv` records, for each paper, the idea taken and how it shaped the project. Prototype missed-diagnosis flags were computed from the predictors present in the 400-patient data (vertigo, no visual symptoms), with an explicit note listing the predictors that are *not* present (sex, bilateral location, pressing character, first-consult clinician) — those are collected by survey Q4, Q5, Q7, Q13 instead.

---

## 7. Results

All numbers below are the actual output of running `analysis.py` (see `outputs/results.txt`). Nothing here is invented.

### 7.1 Patients and symptoms (Part A, real numbers)

- Age: mean 31.7 years, median 28, range 15–77, standard deviation 12.1.
- Nausea: 98.8% of patients, Photophobia: 98.0%, Phonophobia: 97.8%, Visual disturbances: 79.5%, Vomit: 32.2%, Vertigo: 12.5%.
- Almost every patient in this diagnosed cohort has nausea, photophobia and phonophobia — because these are the symptoms doctors asked about to diagnose migraine. That is a property of a *diagnosed* cohort, not of the general population.

### 7.2 Model results (Part A)

| Model | Test Accuracy (n=80) | 5-Fold CV Accuracy |
|---|---|---|
| **Logistic Regression** | **93.8%** | **92.2% (±4.1%)** |
| Gradient Boosting | 92.5% | 90.3% (±3.8%) |
| Random Forest | 88.8% | 89.7% (±3.4%) |

Logistic Regression is best on both test accuracy and cross-validation — and it is also the simplest and most interpretable model, which matters in medicine.

**Per-class results — Logistic Regression (test set, 80 patients):**

| Type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Basilar-type aura | 1.00 | 0.75 | 0.86 | 4 |
| Familial hemiplegic migraine | 0.75 | 0.60 | 0.67 | 5 |
| Migraine without aura | 0.92 | 1.00 | 0.96 | 12 |
| Other | 1.00 | 0.67 | 0.80 | 3 |
| Sporadic hemiplegic migraine | 1.00 | 0.67 | 0.80 | 3 |
| Typical aura with migraine | 0.94 | 1.00 | 0.97 | 49 |
| Typical aura without migraine | 1.00 | 1.00 | 1.00 | 4 |

The two common types are classified almost perfectly (recall 100%). The rare types are the problem: Familial hemiplegic recall is only 60%, and "Other" / Sporadic recall 67% — but each has only 3–5 patients in the test set, so one missed patient drops recall by 20–33 percentage points. That is why support counts are shown next to every score. Random Forest is worse on rare types (Basilar-type recall 25%, precision 0.50).

**What matters most — feature importance (Random Forest):** Visual 0.248, Intensity 0.102, Age 0.085, Vertigo 0.076, Frequency 0.068. Visual symptoms dominate — more than double the next feature. Clinically sensible: aura is defined by visual disturbances, so visual symptoms separate "typical aura" types from "without aura" types.

### 7.3 Evidence across the 10 papers (Part B)

- **Sample sizes:** largest Rabany et al. 2026 (n=53,065), smallest Cardona-Roca et al. 2025 (n=80). The other student surveys: Samir n=5,954, Almomani n=1,276. Clinical samples: Angerhöfer n=1,130, Elmazny n=515, Barbanti n=215 dosed, Faisal n=146, Suwanlaong n=90.
- **Screened prevalence (student surveys only):** Samir 26.1%, Almomani 52.1% — the Almomani figure must not be quoted as population prevalence (94.2%-female convenience sample; meta-analytic student prevalence ~16.1%).
- **Trigger rates (sleep):** Samir 59.7%, Almomani 83.0%, Elmazny 85.6% (under-sleeping). Noise: Samir 47.4%; Elmazny 84.5% is *awareness* of loud noise as a trigger, not incidence — see §3.7.
- **ML AUCs (forecasting, not classification):** Rabany XGBoost 0.893 (accuracy 81.0%), Faisal time-series 0.84, Faisal decision tree 0.59, Faisal foundation model 0.55. AUC and accuracy are different metrics; they are never mixed in this report.
- **Treatment ≥50% responders (completers-only):** Barbanti 69.0% (74/215), Cardona-Roca 65.0% (completers ≥1 year, n=80).
- **Diagnosis gap:** misdiagnosed at first visit 43.3% (Suwanlaong, n=90); diagnosed before referral — men 57.0% vs women 73.8% (Angerhöfer, n=1,130).

### 7.4 Prototype missed-diagnosis flags (Part C)

In the 400 patients, using only the Angerhöfer/Suwanlaong predictors present in this data: vertigo present in 50 patients (12.5%), no visual symptoms in 82 patients (20.5%), both in 5 patients (1.2%). This is a **prototype only** — sex, bilateral location, pressing character and first-consult clinician are not in this dataset, and the code says so. The survey template collects them for future work.

---

## 8. Discussion

### 8.1 Why did Logistic Regression win?

The features are mostly coded categorical / binary symptom indicators, and the classes are largely separated by a few key symptoms (especially Visual). A linear model captures that separation cleanly, while tree ensembles can overfit a small (n=400) dataset — Random Forest's lower CV score (89.7%) and its poor Basilar-type recall (25%) show that.

### 8.2 Why the honest result is 88–94%, not 99.66%

Khan et al. (2024) augmented 400 records to 1,447 and reported 99.66%. If synthetic records are near-copies of real records, the same patient pattern can appear in both train and test — the model is then tested on data it has effectively already seen. That is leakage. This project used no augmentation, a stratified split, and cross-validation, and got 88.8–93.8%. The gap between 93.8% and 99.66% is the leakage gap, not a modelling gap.

### 8.3 What the 10 papers agree on — and what they do not

They agree that sleep and stress are the top triggers, that photophobia is the commonest symptom, that students mostly self-treat and rarely seek care, and that migraine is frequently missed. They disagree — usefully — on whether AI helps or harms: Rabany/Faisal show forecasting can work (AUC 0.84–0.89) when time is modelled properly, while Faisal's own decision tree (AUC 0.59) shows that the same data with the wrong model is near chance. That disagreement is the point of the head-to-head comparison in Part A.

### 8.4 What is new here

No single paper in the set does all three things this project does: (1) honest classification on the real 400-patient data, (2) a quantitative comparison of the 10 papers' numbers with caveats attached, and (3) a fixed study design that removes — by design — the sampling, confirmation, time-series, attrition and honesty failures documented in Section 3. The novelty is the *fixing*, not a new algorithm.

---

## 9. Limitations — Of This Project, Honestly

1. **Small dataset:** 400 patients, from one hospital, in 2013. No external validation on another hospital or country.
2. **Class imbalance:** 61.8% in one class; rare classes have 14–24 patients total. Per-class recall for rare types is unstable (support 3–5 in the test set).
3. **Coded features:** Duration, Frequency, Location, Character etc. are coded numbers, not raw clinical text — some information is lost, and the codebook meanings of some codes are not asserted in this project.
4. **No trigger / lifestyle data in the 400 records:** sleep, stress, caffeine, weather — the top triggers in the literature — are not in this dataset. They are collected only in the proposed diary design.
5. **No prospective test:** the model was not tested on new patients arriving at a clinic. The diary cohort in §4.3 is proposed, not yet run.
6. **The diary design is a design, not data:** its fixes (stratified sampling, clinical confirmation, case-crossover analysis, attrition flow) are specified in §4.3 and in `data/survey_template.csv`, but they have not yet been executed. A future report must report whether they were actually followed.
7. **Not a diagnosis tool:** this classifies *type* among already-diagnosed migraine patients; it does not diagnose migraine vs other headaches, and must not be used for diagnosis.

---

## 10. Conclusion

On 400 real physician-diagnosed migraine patients, a simple Logistic Regression model classified migraine type with 93.8% test accuracy (92.2% cross-validated), beating Random Forest (88.8%) and Gradient Boosting (92.5%). Visual symptoms were the most important feature. Across the 10 papers, the strongest evidence is Rabany's forecasting (AUC 0.893, n=53,065), Samir's student survey (n=5,954), Faisal's time-series comparison (AUC 0.84 vs 0.59), and Angerhöfer's diagnosis-gap finding (men 57.0% vs women 73.8% diagnosed before referral) — and the weakest is Khan's 99.66%, which this project does not claim. The contribution of this project is not a higher accuracy, but a **fixed version**: the same questions as the 10 papers, with their sampling, confirmation, time-series, attrition and honesty failures removed by design. The next step is not a bigger model, but the diary cohort in §4.3 — bigger, more diverse, prospective data, especially for rare migraine types, for men and atypical presentations, and for students in South Asia, where primary data are barely represented.

---

## 11. How to Run / Files

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

| File | What it is |
|---|---|
| `analysis.py` | Full real code: Part A classification, Part B papers analysis, Part C idea map + missed-diagnosis flags |
| `data/migraine.csv` | Real dataset, 400 rows × 24 columns |
| `data/papers.csv` | Published numbers from the 10 papers, with URLs |
| `data/survey_template.csv` | 15-question student survey (Q1–Q15), each question tagged with its source paper idea |
| `outputs/type_distribution.png` | Migraine types in the 400 patients |
| `outputs/symptom_rates.png` | Most common symptoms |
| `outputs/age_distribution.png` | Age distribution (mean 31.7, median 28) |
| `outputs/feature_importance.png` | Random Forest feature importance (Visual 0.248) |
| `outputs/model_comparison.png` | Test vs CV accuracy for the 3 models |
| `outputs/paper_sample_sizes.png` | Sample sizes across the 10 papers (log scale) |
| `outputs/trigger_rates.png` | Sleep/noise trigger rates across papers |
| `outputs/paper_ml_aucs.png` | Forecasting AUCs in the papers |
| `outputs/model_comparison.csv` | Model accuracies as a table |
| `outputs/idea_map.csv` | One idea from each paper + how it shaped the project |
| `outputs/results.txt` | Full printed output from running `analysis.py` |
| `README.md` | Project summary for GitHub |
| `requirements.txt` | pandas, numpy, matplotlib, scikit-learn |

---

## 12. References

1. Rabany, L. et al. (2026). Machine Learning Assessment of Next-Day Migraine Likelihood Using Data From 53,000 App Users Living With Migraine. *Neurology Open Access* 2(3):e000144. https://www.neurology.org/doi/10.1212/WN9.0000000000000144?cookieSet=1
2. Samir, A. A. et al. (2025). Burden of migraine among university students in the Middle East and North Africa. *BMC Public Health* 25:3605. https://pubmed.ncbi.nlm.nih.gov/41146134/
3. Faisal, F. et al. (2026). Forecasting migraine with time-series machine learning from mobile health data. *Journal of Headache and Pain* 27:91. https://pubmed.ncbi.nlm.nih.gov/41913100/
4. Angerhöfer, C. et al. (2025). Sex-related disparities in migraine recognition and management. *Journal of Headache and Pain* 26:225. https://link.springer.com/article/10.1186/s10194-025-02189-8
5. Barbanti, P. et al. (2025). A 24-week prospective, multicenter, real-world study on eptinezumab's effectiveness and safety in migraine prevention (EMBRACE II). *Journal of Neurology* 272:382. https://pubmed.ncbi.nlm.nih.gov/40332560/
6. Elmazny, A. et al. (2025). Migraine triggers and lifestyle modifications: an assessment of patients' awareness and the role of healthcare providers in patient education. *Journal of Headache and Pain* 26:189. https://pmc.ncbi.nlm.nih.gov/articles/PMC12400567/
7. Almomani, M. A. et al. (2025). Migraine Headache and Its Related Disability Factors Among University Students in Jordan. *Journal of Pain Research* 18:2093–2104. https://pmc.ncbi.nlm.nih.gov/articles/PMC12009572/
8. Cardona-Roca, L. et al. (2025). Impact of anti-CGRP monoclonal antibodies on treatment satisfaction and quality of life in patients with resistant migraine. *BMC Neurology* 25:418. https://doi.org/10.1186/s12883-025-04384-1
9. Suwanlaong, K. et al. (2025). Factors contributing to inaccurate migraine diagnosis. *Journal of Headache and Pain* 27:35. https://pubmed.ncbi.nlm.nih.gov/41449358/
10. Khan, L. et al. (2024). Migraine headache (MH) classification using machine learning methods with data augmentation. *Scientific Reports* 14:5180. https://pmc.ncbi.nlm.nih.gov/articles/PMC10908834/ — *cite with caution, see §2.10 and §3.10*
11. Sanchez-Sanchez, P. A. et al. (2020). Migraine Classification Model, Code Ocean. https://doi.org/10.24433/CO.2826453.v1 — *dataset source*
12. De la Hoz, E. et al. (2014). Analysis of Artificial Neural Network Models, for a Migraine Diagnosis System with Aura and without Aura. Master's thesis, Hospital Materno Infantil de Soledad. — *original dataset*

---

## 13. Viva Q&A

**Q: Where did the data come from? Is it real?**
A: Yes, real. 400 patients diagnosed by a physician at Hospital Materno Infantil de Soledad, Colombia, 2013. Published in F1000Research / Code Ocean and on Kaggle. No synthetic data was used in this project.

**Q: What is new about this project — is it not just the same as Khan et al.?**
A: No. Khan et al. augmented 400 records to 1,447 and claimed 99.66%. This project does the opposite: no augmentation, stratified split, 5-fold CV, per-class reporting — and then adds a second part (the 10 papers' evidence, compared with caveats) and a third part (a fixed diary study design that removes the papers' sampling, confirmation and attrition failures). The novelty is the fixing, documented paper-by-paper in Section 3.

**Q: Why is Logistic Regression the best model?**
A: Test accuracy 93.8%, CV 92.2% — best on both. The classes are largely separated by a few key symptoms (especially Visual), which a linear model captures cleanly. It is also interpretable, which matters in medicine.

**Q: Why not 99.66% like Khan et al.?**
A: Khan et al. augmented 400 → 1,447, likely leakage. Honest result is 88–94%. Their paper also has an internal inconsistency: Random Forest 98.5% in the abstract vs 88.5% in Table 3.

**Q: What is the biggest weakness?**
A: Class imbalance and small data for rare types. Familial hemiplegic migraine has only 24 patients in total, 5 in the test set — recall 60% there. And the data are from one hospital in 2013, with no external validation.

**Q: What does Visual importance 0.248 mean?**
A: In the Random Forest, Visual symptoms contribute 24.8% of the model's splitting power — more than double the next feature (Intensity, 0.102). Clinically, aura is defined by visual disturbances, so this makes sense.

**Q: How does this project fix Samir/Almomani's sampling problem?**
A: The proposed diary cohort stratifies by sex and faculty instead of convenience-only social media, screens with validated items and confirms a subsample against ICHD-3, and reports "screened-positive", never "diagnosed".

**Q: How does it fix the trigger problem (cross-sectional correlation)?**
A: With a within-person case-crossover analysis in the diary design — compare trigger exposure on days before an attack vs the same student's attack-free days — instead of comparing different students at one time point.

**Q: Can this diagnose migraine?**
A: No. It classifies *type* among patients already diagnosed with migraine. It does not distinguish migraine from tension headache or sinusitis — the commonest misdiagnoses in the literature. A doctor must diagnose.

**Q: How would you improve it next?**
A: Run the diary cohort in §4.3 (150–300 students, 4–8 weeks), add trigger/lifestyle data, test prospectively on new clinic patients, and validate the model on another hospital's data — especially for rare types, men and atypical presentations, and students in South Asia.
