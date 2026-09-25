"""
clean_data.py
Load, validate, and clean the ACS 2024 1-Year PUMS person file.

Run from the project root after downloading the data:
    python src/clean_data.py data/psam_pXX.csv

Output:
    data/pums_clean.csv      cleaned dataset used by the Streamlit app
    data/cleaning_log.txt    record of every cleaning step (useful for your report)

Code meanings follow the 2024 PUMS Data Dictionary. Check each mapping
against the dictionary so you can explain it in your report.
"""

import os
import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. Columns to keep and how to read them
# ------------------------------------------------------------------
COLS = ["SERIALNO", "SPORDER", "PUMA", "AGEP", "SEX", "RAC1P", "HISP", "SCHL",
        "MAR", "MIL", "ESR", "DIS", "DEAR", "DEYE", "DREM", "DPHY", "DDRS",
        "DOUT", "WKHP", "WAGP", "ADJINC", "PWGTP"]

DTYPES = {"SERIALNO": str, "PUMA": str}   # keep letters and leading zeros

# Expected codes, used to catch unexpected values
VALID_CODES = {
    "SEX": {1, 2},
    "RAC1P": set(range(1, 10)),
    "SCHL": set(range(1, 25)),
    "MAR": set(range(1, 6)),
    "MIL": set(range(1, 5)),
    "ESR": set(range(1, 7)),
    "DIS": {1, 2},
    "DEAR": {1, 2}, "DEYE": {1, 2}, "DREM": {1, 2},
    "DPHY": {1, 2}, "DDRS": {1, 2}, "DOUT": {1, 2},
}

DISABILITY_TYPES = {
    "DEAR": "hearing", "DEYE": "vision", "DREM": "cognitive",
    "DPHY": "ambulatory", "DDRS": "self_care", "DOUT": "independent_living",
}

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


# ------------------------------------------------------------------
# 2. Load
# ------------------------------------------------------------------
def load(path):
    if path.lower().endswith(".sas7bdat"):
        # SAS version: codes are stored as text (e.g. "01"), blanks as "".
        df = pd.read_sas(path, encoding="latin1")[COLS]
        for col in COLS:
            if col not in DTYPES:
                df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
    else:
        df = pd.read_csv(path, usecols=COLS, dtype=DTYPES)
    df["PWGTP"] = df["PWGTP"].astype(int)
    log(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {path}")
    return df


# ------------------------------------------------------------------
# 3. Validate structure
# ------------------------------------------------------------------
def validate(df):
    # Each person is uniquely identified by household serial number + person number
    dupes = df.duplicated(subset=["SERIALNO", "SPORDER"]).sum()
    log(f"Duplicate person records: {dupes}")
    if dupes:
        df = df.drop_duplicates(subset=["SERIALNO", "SPORDER"])
        log(f"  Removed {dupes} exact duplicate person IDs")

    # Codes outside the documented range
    for col, valid in VALID_CODES.items():
        bad = df[col].notna() & ~df[col].isin(valid)
        if bad.sum():
            log(f"  WARNING: {col} has {bad.sum()} unexpected values: "
                f"{sorted(df.loc[bad, col].unique())[:10]}")
    return df


# ------------------------------------------------------------------
# 4. Filter to the study population
# ------------------------------------------------------------------
def filter_working_age(df):
    before = len(df)
    df = df[df["AGEP"].between(18, 64)].copy()
    log(f"Filtered to ages 18-64: {before:,} -> {len(df):,} rows "
        f"(study population is working-age adults, not a data-quality drop)")
    return df


# ------------------------------------------------------------------
# 5. Missing values: separate 'not applicable' from truly missing
# ------------------------------------------------------------------
def report_missing(df, title):
    log(f"\nMissing values ({title}):")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        log("  none")
    for col, n in miss.items():
        log(f"  {col}: {n:,} ({n / len(df):.1%})")


def handle_missing(df):
    # WKHP (usual hours worked per week) is blank when the person did not
    # work in the past 12 months. That is "not applicable", so it becomes 0
    # and a flag keeps the information.
    df["worked_past_year"] = df["WKHP"].notna().astype(int)
    df["WKHP"] = df["WKHP"].fillna(0)
    log("\nWKHP blanks = did not work in past 12 months -> set to 0, "
        "added flag 'worked_past_year'")

    # MIL is blank only for people under 17, who were filtered out.
    # Any remaining blanks in key columns are reported, not silently dropped.
    for col in ["ESR", "DIS", "SCHL", "MIL"]:
        n = df[col].isna().sum()
        if n:
            log(f"  WARNING: {col} still has {n} blanks - investigate before modeling")
    return df


# ------------------------------------------------------------------
# 6. Adjust income to 2024 dollars
# ------------------------------------------------------------------
def adjust_income(df):
    # ADJINC is stored with 6 implied decimal places
    df["wages_adj"] = df["WAGP"] * df["ADJINC"] / 1_000_000
    log("Created 'wages_adj' = WAGP x ADJINC / 1,000,000 (inflation-adjusted wages)")
    return df


# ------------------------------------------------------------------
# 7. Decode codes into readable labels and engineer features
# ------------------------------------------------------------------
def engineer(df):
    # Target label: 1-2 civilian employed, 4-5 armed forces employed,
    # 3 unemployed, 6 not in labor force
    df["employed"] = df["ESR"].isin([1, 2, 4, 5]).astype(int)
    df["labor_status"] = df["ESR"].map({
        1: "Employed", 2: "Employed", 4: "Employed", 5: "Employed",
        3: "Unemployed", 6: "Not in labor force"})

    df["has_disability"] = (df["DIS"] == 1).astype(int)
    for code, name in DISABILITY_TYPES.items():
        df[f"dis_{name}"] = (df[code] == 1).astype(int)
    df["num_disabilities"] = df[[f"dis_{n}" for n in DISABILITY_TYPES.values()]].sum(axis=1)

    df["sex"] = df["SEX"].map({1: "Male", 2: "Female"})

    df["race_eth"] = np.where(
        df["HISP"] != 1, "Hispanic",
        df["RAC1P"].map({1: "White", 2: "Black", 3: "Native American",
                         4: "Native American", 5: "Native American",
                         6: "Asian", 7: "Pacific Islander",
                         8: "Other", 9: "Two or more"}))

    df["education"] = pd.cut(
        df["SCHL"], bins=[0, 15, 17, 19, 20, 21, 24],
        labels=["Less than HS", "HS diploma/GED", "Some college",
                "Associate", "Bachelor's", "Graduate degree"])

    df["marital"] = df["MAR"].map({1: "Married", 2: "Widowed", 3: "Divorced",
                                   4: "Separated", 5: "Never married"})

    df["veteran"] = (df["MIL"] == 2).astype(int)

    df["age_group"] = pd.cut(df["AGEP"], bins=[17, 24, 34, 44, 54, 64],
                             labels=["18-24", "25-34", "35-44", "45-54", "55-64"])
    log("Decoded categorical codes and created features: employed, labor_status, "
        "has_disability, dis_* flags, num_disabilities, race_eth, education, "
        "marital, veteran, age_group")
    return df


# ------------------------------------------------------------------
# 8. Flag outliers (flag, do not drop)
# ------------------------------------------------------------------
def flag_outliers(df):
    for col in ["wages_adj", "WKHP"]:
        # Only people with positive values, since zeros are "did not work"
        vals = df.loc[df[col] > 0, col]
        q1, q3 = vals.quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        flag = f"outlier_{col.lower()}"
        df[flag] = (df[col] > upper).astype(int)
        log(f"{col}: IQR upper fence = {upper:,.1f}; "
            f"{df[flag].sum():,} records flagged in '{flag}' (kept, not dropped)")
    log("Note: Census top-codes very high wages, so extreme values are capped "
        "in the source data.")
    return df


# ------------------------------------------------------------------
# 9. Count people: survey sample and weighted population estimate
# ------------------------------------------------------------------
def summarize_counts(df):
    # "People surveyed" = rows in the file.
    # "Estimated population" = sum of person weights (PWGTP), which is how
    # many real people each surveyed person represents.
    log("\nNumber of people (ages 18-64):")
    log(f"  Surveyed: {len(df):,}   Estimated population: {df['PWGTP'].sum():,}")

    table = (df.groupby(["has_disability", "labor_status"])
               .agg(surveyed=("PWGTP", "size"), est_population=("PWGTP", "sum"))
               .reset_index())
    table["has_disability"] = table["has_disability"].map({1: "With disability",
                                                           0: "Without disability"})
    log("\n" + table.to_string(index=False))

    log("\nPeople with each disability type (a person can have more than one):")
    for name in DISABILITY_TYPES.values():
        sub = df[df[f"dis_{name}"] == 1]
        log(f"  {name:<20} surveyed: {len(sub):>7,}   "
            f"est. population: {sub['PWGTP'].sum():>10,}")

    # Weighted employment rates (use these in your report)
    for flag, label in [(1, "with disability"), (0, "without disability")]:
        sub = df[df["has_disability"] == flag]
        rate = (sub["employed"] * sub["PWGTP"]).sum() / sub["PWGTP"].sum()
        log(f"Weighted employment rate, {label}: {rate:.1%}")
    table.to_csv("data/people_counts.csv", index=False)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(path):
    df = load(path)
    df = validate(df)
    df = filter_working_age(df)
    report_missing(df, "before cleaning")
    df = handle_missing(df)
    df = adjust_income(df)
    df = engineer(df)
    df = flag_outliers(df)
    report_missing(df, "after cleaning")

    log(f"\nFinal dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    summarize_counts(df)

    df.to_csv("data/pums_clean.csv", index=False)
    with open("data/cleaning_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    print("\nSaved data/pums_clean.csv and data/cleaning_log.txt")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main(r"C:\Users\Brandon\Downloads\csv_pnc\psam_p37.csv")
