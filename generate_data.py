"""
Synthetic data generator for the PredictAFR demo.

Produces two CSVs consumed by streamlit_app.py:
  - patients.csv      : one row per patient, all clinical variables
  - shap_values.csv   : long-format per-patient feature attributions that are
                        internally consistent with each patient's displayed
                        risk score (baseline + sum(contributions) = risk)

This is NOT a real predictive model. Weights below are chosen to produce
clinically *plausible* directions of effect (e.g. larger LA volume increases
risk, preserved EF is protective) for teaching / demo purposes only.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_PATIENTS = 60
BASELINE_RISK = 0.40  # cohort baseline 12-month recurrence risk

rng = np.random.default_rng(RNG_SEED)


def sample_categorical(n, categories, p=None):
    return rng.choice(categories, size=n, p=p)


def bernoulli(n, p):
    return (rng.random(n) < p).astype(int)


def build_patient_cohort(n=N_PATIENTS):
    patient_id = np.arange(101, 101 + n)

    # --- Demographics ---
    age = rng.integers(45, 86, n)
    sex = sample_categorical(n, ["Male", "Female"], p=[0.62, 0.38])
    bmi = np.round(rng.normal(29, 4.5, n).clip(19, 45), 1)
    alcohol_use = sample_categorical(n, ["No current", "Occasional", "Regular"], p=[0.35, 0.45, 0.20])
    smoking_status = sample_categorical(n, ["Never", "Ex-smoker", "Current"], p=[0.5, 0.4, 0.1])
    drug_use = sample_categorical(n, ["Never", "Ex-user", "Current"], p=[0.85, 0.12, 0.03])

    # --- AF details (drawn before other variables so we can correlate) ---
    af_type = sample_categorical(
        n, ["Paroxysmal", "Persistent", "Long-standing persistent"], p=[0.5, 0.35, 0.15]
    )
    persistent_like = np.isin(af_type, ["Persistent", "Long-standing persistent"]).astype(int)
    af_duration_months = np.round(
        rng.gamma(shape=2.0, scale=(10 + persistent_like * 18), size=n)
    ).clip(1, 180).astype(int)
    concomitant_flutter = bernoulli(n, 0.18 + 0.1 * persistent_like)
    prior_cardioversion = bernoulli(n, 0.25 + 0.2 * persistent_like)
    ablation_type = sample_categorical(n, ["Radiofrequency", "Cryoballoon"], p=[0.6, 0.4])

    # --- Clinical comorbidities ---
    hypertension = bernoulli(n, 0.55)
    chf = bernoulli(n, 0.15 + 0.1 * persistent_like)
    vascular_disease = bernoulli(n, 0.20)
    thyroid_disorder = bernoulli(n, 0.12)
    renal_disorder = bernoulli(n, 0.10)
    hepatic_disorder = bernoulli(n, 0.06)
    respiratory_disorder = bernoulli(n, 0.14)
    osa = bernoulli(n, 0.28)
    diabetes = bernoulli(n, 0.22)
    prior_stroke_tia_te = bernoulli(n, 0.12)
    prior_mi = bernoulli(n, 0.10)

    # --- Risk scores derived from age/sex/comorbidities ---
    chads2vasc = (
        1 * (age >= 65).astype(int)
        + 1 * (age >= 75).astype(int)
        + 1 * (sex == "Female").astype(int)
        + 1 * chf
        + 1 * hypertension
        + 2 * prior_stroke_tia_te
        + 1 * vascular_disease
        + 1 * diabetes
    ).clip(0, 9)

    hasbled = (
        1 * hypertension
        + 1 * renal_disorder
        + 1 * hepatic_disorder
        + 1 * prior_stroke_tia_te
        + 1 * bernoulli(n, 0.08)  # bleeding history
        + 1 * (age > 75).astype(int)
        + 1 * bernoulli(n, 0.10)  # labile INR / drugs predisposing to bleeding
    ).clip(0, 9)

    # --- Medications ---
    oac_use = bernoulli(n, 0.15 + 0.6 * (chads2vasc >= 2))
    antiplatelet_use = bernoulli(n, 0.15)
    antiarrhythmic_use = bernoulli(n, 0.55 + 0.15 * persistent_like).clip(0, 1)
    beta_blocker_use = bernoulli(n, 0.60)
    statin_use = bernoulli(n, 0.45 + 0.15 * hypertension)

    # --- Biochemistry ---
    haemoglobin = np.round(
        np.where(sex == "Male", rng.normal(148, 12, n), rng.normal(133, 10, n)), 0
    ).clip(90, 180)
    creatinine = np.round(rng.normal(85, 20, n) + renal_disorder * 25, 0).clip(45, 220)
    egfr_over_90 = ((creatinine < 80) & (renal_disorder == 0) & (age < 70)).astype(int)
    alt = np.round(rng.normal(28, 12, n) + hepatic_disorder * 20, 0).clip(8, 120)

    # --- Imaging ---
    ivsd_cm = np.round(rng.normal(1.05, 0.13, n).clip(0.7, 1.6), 2)
    lvvi = np.round(rng.normal(55, 10, n).clip(30, 90), 1)
    lvef = np.round((rng.normal(58, 7, n) - chf * 10 - prior_mi * 6).clip(25, 70), 0)
    lavi = np.round(
        (rng.normal(32, 8, n) + persistent_like * 10 + af_duration_months * 0.05).clip(20, 70), 1
    )
    e_a_ratio = np.round(rng.normal(1.1, 0.35, n).clip(0.4, 2.6), 2)
    e_e_prime = np.round((rng.normal(10, 3, n) + persistent_like * 2).clip(4, 22), 1)

    # --- Left atrial strain (all inversely related to LAVi / AF chronicity) ---
    la_reservoir_strain = np.round(
        (rng.normal(32, 8, n) - persistent_like * 8 - (lavi - 32) * 0.25).clip(8, 50), 1
    )
    la_conduit_strain = np.round(
        (rng.normal(14, 5, n) - persistent_like * 4 - (lavi - 32) * 0.1).clip(2, 28), 1
    )
    la_contractile_strain = np.round(
        (rng.normal(13, 4, n) - persistent_like * 5 - (lavi - 32) * 0.12).clip(1, 24), 1
    )
    time_to_peak_ms = np.round(rng.normal(360, 30, n) + persistent_like * 15, 0).clip(280, 460)

    # --- APPLE score (Age>65, Persistent AF, imPaired eGFR<60, LA diameter>=45mm-proxy via LAVi>=42, EF<=50%) ---
    apple_score = (
        (age > 65).astype(int)
        + persistent_like
        + (egfr_over_90 == 0).astype(int)
        + (lavi >= 42).astype(int)
        + (lvef <= 50).astype(int)
    )

    df = pd.DataFrame(
        {
            "PatientID": patient_id,
            "Age": age,
            "Sex": sex,
            "BMI": bmi,
            "Alcohol_Use": alcohol_use,
            "Smoking_Status": smoking_status,
            "Drug_Use": drug_use,
            "APPLE_Score": apple_score,
            "CHA2DS2_VASc": chads2vasc,
            "HAS_BLED": hasbled,
            "CHF": chf,
            "Hypertension": hypertension,
            "Vascular_Disease": vascular_disease,
            "Thyroid_Disorder": thyroid_disorder,
            "Renal_Disorder": renal_disorder,
            "Hepatic_Disorder": hepatic_disorder,
            "Respiratory_Disorder": respiratory_disorder,
            "OSA": osa,
            "Diabetes": diabetes,
            "Prior_Stroke_TIA_TE": prior_stroke_tia_te,
            "Prior_MI": prior_mi,
            "OAC_Use": oac_use,
            "Antiplatelet_Use": antiplatelet_use,
            "Antiarrhythmic_Use": antiarrhythmic_use,
            "Beta_Blocker_Use": beta_blocker_use,
            "Statin_Use": statin_use,
            "Haemoglobin": haemoglobin,
            "Creatinine": creatinine,
            "eGFR_Over_90": egfr_over_90,
            "ALT": alt,
            "AF_Type": af_type,
            "AF_Duration_Months": af_duration_months,
            "Concomitant_Flutter": concomitant_flutter,
            "Prior_Cardioversion": prior_cardioversion,
            "Ablation_Type": ablation_type,
            "IVSd": ivsd_cm,
            "LVVi": lvvi,
            "LVEF": lvef,
            "LAVi": lavi,
            "E_A_Ratio": e_a_ratio,
            "E_e_prime_avg": e_e_prime,
            "LA_Reservoir_Strain": la_reservoir_strain,
            "LA_Conduit_Strain": la_conduit_strain,
            "LA_Contractile_Strain": la_contractile_strain,
            "Time_to_Peak_ms": time_to_peak_ms,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Mock "explainability" model: additive contributions in probability space.
# Continuous features contribute weight * z-score; binary features contribute
# a fixed step. Contributions are built so that
#     BASELINE_RISK + sum(contributions) == final risk score
# for every patient, giving a SHAP-style waterfall that is internally
# consistent with the headline number shown elsewhere in the app.
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS = {
    # feature_column, display name, weight, direction handled by sign of weight
    "LAVi": ("LA Volume Index (LAVi)", 0.014),
    "persistent_like": ("Persistent / Long-standing AF", 0.06),
    "LA_Reservoir_Strain": ("LA Reservoir Strain", -0.010),
    "AF_Duration_Months": ("AF Duration", 0.0035),
    "LVEF": ("Left Ventricular EF", -0.006),
    "Age": ("Age", 0.0025),
    "CHA2DS2_VASc": ("CHA2DS2-VASc Score", 0.010),
    "E_e_prime_avg": ("E/e' Average (Diastolic Function)", 0.006),
    "Hypertension": ("Hypertension", 0.03),
    "OSA": ("Obstructive Sleep Apnoea", 0.035),
    "Diabetes": ("Diabetes Mellitus", 0.025),
    "Antiarrhythmic_Use": ("Antiarrhythmic Therapy", -0.04),
    "Prior_Cardioversion": ("Prior DC Cardioversion", 0.02),
    "Concomitant_Flutter": ("Concomitant Atrial Flutter", 0.02),
    "eGFR_Over_90": ("Preserved Renal Function (eGFR>90)", -0.025),
    "LA_Contractile_Strain": ("LA Contractile Strain", -0.012),
    "Beta_Blocker_Use": ("Beta Blocker Therapy", -0.015),
}

BINARY_FEATURES = {
    "persistent_like", "Hypertension", "OSA", "Diabetes", "Antiarrhythmic_Use",
    "Prior_Cardioversion", "Concomitant_Flutter", "eGFR_Over_90", "Beta_Blocker_Use",
}


def build_shap_table(patients: pd.DataFrame) -> pd.DataFrame:
    patients = patients.copy()
    patients["persistent_like"] = patients["AF_Type"].isin(
        ["Persistent", "Long-standing persistent"]
    ).astype(int)

    # z-score continuous features across the cohort for standardized contributions
    zscores = {}
    for col in FEATURE_WEIGHTS:
        if col in BINARY_FEATURES:
            continue
        mu, sigma = patients[col].mean(), patients[col].std(ddof=0)
        zscores[col] = (patients[col] - mu) / (sigma if sigma > 0 else 1)

    records = []
    for _, row in patients.iterrows():
        contributions = {}
        for col, (label, weight) in FEATURE_WEIGHTS.items():
            if col in BINARY_FEATURES:
                contributions[label] = weight * row[col]
            else:
                contributions[label] = weight * zscores[col].loc[row.name]

        raw_total = sum(contributions.values())
        # small amount of independent noise so risk isn't a pure linear function
        noise = rng.normal(0, 0.015)
        final_risk = float(np.clip(BASELINE_RISK + raw_total + noise, 0.05, 0.95))

        # rescale contributions proportionally so they still sum exactly to
        # (final_risk - BASELINE_RISK), keeping the waterfall internally consistent
        adjustment = (final_risk - BASELINE_RISK) - raw_total
        n_features = len(contributions)
        for label in contributions:
            contributions[label] += adjustment / n_features

        for label, value in contributions.items():
            records.append(
                {
                    "PatientID": row["PatientID"],
                    "Feature": label,
                    "SHAP_Value": round(value, 4),
                    "Type": "Risk Factor" if value >= 0 else "Protective Factor",
                }
            )

        # stash final risk + outcome on the patients frame via side channel
        patients.loc[row.name, "Risk_Score"] = round(final_risk, 3)

    outcome_prob = patients["Risk_Score"]
    patients["Outcome"] = np.where(
        bernoulli(len(patients), outcome_prob.values), "Recurred", "Sinus Rhythm"
    )

    shap_df = pd.DataFrame(records)
    return patients, shap_df


if __name__ == "__main__":
    patients_df = build_patient_cohort()
    patients_df, shap_df = build_shap_table(patients_df)
    patients_df.drop(columns=["persistent_like"], errors="ignore", inplace=True)

    patients_df.to_csv("patients.csv", index=False)
    shap_df.to_csv("shap_values.csv", index=False)

    print(patients_df.shape, shap_df.shape)
    print(patients_df[["PatientID", "AF_Type", "LAVi", "Risk_Score", "Outcome"]].head(10))
