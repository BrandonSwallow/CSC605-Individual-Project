# Disability and Employment in North Carolina

**CSC 605 Data Science · Individual Project · Fall 2026 · Brandon Swallow**

An interactive Streamlit report analyzing the employment gap between working-age adults with and
without disabilities in North Carolina, using the U.S. Census Bureau's 2024 American Community Survey
Public Use Microdata Sample (PUMS). The app integrates an LLM through LangChain and the Google Gemini
API (free tier) to support and document major analysis decisions.

## Dataset source

* **American Community Survey (ACS) 2024 1-Year PUMS, person file, North Carolina** (U.S. Census Bureau, public domain)
* Download folder: https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/ (file `sas_pnc.zip`)
* Data.gov listing: https://catalog.data.gov/dataset/2024-american-community-survey-1-year-estimates-public-use-microdata-sample
* Documentation and data dictionary: https://www.census.gov/programs-surveys/acs/microdata/documentation.html

Raw file: 114,270 people × 286 columns. Study population: 65,915 adults aged 18-64, representing
about 6.72 million North Carolinians.

## Project objective

1. Measure how large the employment gap is between adults (18-64) with and without disabilities, and
   whether it differs by disability type (hearing, vision, cognitive, ambulatory, self-care,
   independent living).
2. Identify which factors (education, age, sex, race/ethnicity, marital status, veteran status,
   disability type and number of disabilities) best predict employment among adults with disabilities.

Methods: survey-weighted estimates with replicate-weight margins of error; chi-square tests (with a
design-effect correction), Mann-Whitney U, Spearman correlation; Logistic Regression (baseline),
Random Forest and Gradient Boosting (advanced).

## Key findings

* Employment rate: **44.3%** with a disability vs **79.3%** without, a **35.0-point gap** (90% CI 33.8-36.2).
* Most of the gap is labor-force non-participation (52% vs 18%), not unemployment.
* Employment ranges from 77% (hearing difficulty only) to 26% (two or more disability types).
* Best model: Random Forest, test ROC-AUC 0.818 (Logistic Regression 0.801). Top predictors: age,
  education, independent-living difficulty.

## Repository structure

```
data/
  raw/sas_pnc.zip            original Census download (NC person file)
  pums_clean.csv             cleaned analysis dataset (65,915 rows x 42 columns)
  pums_repweights.csv.gz     80 replicate weights per person (for standard errors)
  missing_audit.csv          blank values before/after the age filter
  cleaning_log.txt           record of every cleaning step
src/
  clean_data.py              load, validate, clean, engineer features
  analysis.py                weighted estimates, tests, models, figures
  ai_assistant.py            LangChain + Gemini API integration, AI decision log
  test_gemini.py             checks that your Gemini key works
  make_figures.py            regenerate all figures and result tables
  app.py                     Streamlit report
figures/                     generated figures (fig01-fig14, numbered as in the report)
report/
  tables/                    result tables (CSV)
  ai_log.json                saved AI prompts, responses and my evaluations (created by the app)
  final_report.pdf           PDF export of the Streamlit report
docs/                        project proposal
```

## How to run

Requires Python 3.11 or newer.

```bash
git clone https://github.com/BrandonSwallow/CSC605-Individual-Project.git
cd CSC605-Individual-Project
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/clean_data.py      # optional: rebuilds data/ from data/raw/sas_pnc.zip
python src/make_figures.py    # optional: rebuilds figures/ and report/tables/
streamlit run src/app.py      # opens the report at http://localhost:8501
```

The cleaned data is already committed, so `streamlit run src/app.py` works right after installing
requirements. The first load trains the models (about one minute); later reloads are cached.

### AI assistant setup (Google Gemini free tier)

The app uses LangChain's `ChatGoogleGenerativeAI` with the **free** Gemini API. No billing is needed.

1. Create a free key at https://aistudio.google.com/apikey (sign in with a Google account → **Create API key**).
2. Give the key to the app in **one** of these ways:
   * copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste the key there
     (this file is git-ignored, so the key is never committed), or
   * set an environment variable: `$env:GOOGLE_API_KEY = "..."` (PowerShell) or
     `export GOOGLE_API_KEY="..."` (macOS/Linux), or
   * paste it into the sidebar while the app is running.
3. Optional check: `python src/test_gemini.py` lists the models your key can use and sends one test prompt.
4. In Section 7 of the app, click **Ask the AI** for each decision, read the answer, choose
   Accepted / Modified / Rejected, write a justification, and click **Save evaluation**. Everything is
   stored in `report/ai_log.json` and shown in the report.

**How the free tier is handled**

* The sidebar checks the key and lists only the free-tier models it can use, best first:
  `gemini-3.5-flash`, `gemini-3-flash-preview`, `gemini-2.5-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-flash-lite`
  (the free models on Google's pricing page as of November 2026).
* Free keys are limited to a few requests per minute and a daily cap per model. If a model is out of
  quota or unavailable, the app automatically retries with the next free model and shows which model
  answered. Each button click is one request, so the 7 decision prompts plus a few chat questions fit
  comfortably within the free limits.
* Gemini 3 models use their default temperature (Google advises against lowering it); 2.5 models use 0.2.
* Google may use free-tier prompts to improve its products. The prompts contain only public Census
  results.
* If Google renames models, choose **Other...** in the sidebar and type any model name your key lists.

OpenAI is also supported (choose it in the sidebar and use an `OPENAI_API_KEY`), but it requires a paid account.

### Exporting the PDF report

With the app open and all sections loaded, use the browser's print dialog (Ctrl+P / Cmd+P, or the
Streamlit menu ⋮ → Print), choose **Save as PDF**, and enable **Background graphics**. Save it as
`report/final_report.pdf`. The sidebar and buttons are hidden automatically when printing.

## Use of survey weights

* Population estimates (rates, counts, medians, charts) use the person weight `PWGTP`.
* Standard errors and 90% margins of error use the 80 replicate weights (successive-difference
  replication: SE = sqrt(4/80 × Σ(θᵣ − θ)²)).
* Chi-square tests are reported on sample counts and with a first-order Rao-Scott design-effect correction.
* Models are trained unweighted; a weighted training run is compared in Section 6.1 and gives
  nearly identical results.
