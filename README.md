# Predictive Maintenance Capstone — Final Project Repository

This repository contains the existing Predictive Maintenance capstone work, extended with a Streamlit interface and dependencies needed by the training workflow.

## Project contents

- `data/engine_data.csv` — source dataset used by the project.
- `scripts/validate_data.py` — validates expected columns and prints a data summary.
- `scripts/prepare_data.py` — creates reproducible stratified train/test CSV files.
- `scripts/train_model.py` — compares six classifiers with cross-validated GridSearchCV and saves the best pipeline.
- `models/best_model.joblib` — saved model pipeline used by the app.
- `app.py` — Streamlit prediction interface.
- `.github/workflows/ml_pipeline.yml` — two-job GitHub Actions pipeline (data preparation artifact, model training, evaluation, model commit).
- `notebooks/` and `figures/` — project analysis and supporting outputs.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_data.py
python scripts/prepare_data.py
python scripts/train_model.py
streamlit run app.py
```

## Publish to GitHub

1. Create a new **public** repository named `predictive-maintenance-capstone` (do not initialize it with a README, license, or .gitignore).
2. Extract this ZIP and open a terminal in the extracted `Predictive_Maintenance_Capstone` folder.
3. Run:

```bash
git init
git add .
git commit -m "Add predictive maintenance capstone project"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/predictive-maintenance-capstone.git
git push -u origin main
```

4. In GitHub, open **Settings → Actions → General → Workflow permissions** and enable **Read and write permissions**. Save.
5. Open **Actions**, select **Predictive Maintenance ML Pipeline**, and confirm the workflow completes. The workflow is configured to commit the newly trained model and evaluation output to `main`.

If branch protection rules prevent the bot from pushing, adjust the rule or use a pull-request workflow instead. Do not report the workflow as successful until a real run is green.

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with the GitHub account that owns the repository.
2. Choose **Create app**, select this repository, branch `main`, and file `app.py`.
3. Deploy and wait for the app health check to pass.
4. Record the actual public app URL and capture a screenshot showing the running app and a sample prediction.

## Important target-label limitation

The source dataset uses numeric target labels 0 and 1. The operational mapping of those labels (for example, normal versus failure) has not been independently confirmed in the supplied project materials. The app therefore displays **Predicted recorded class** and must not be represented as a validated failure alert until the dataset owner or authoritative documentation confirms the mapping.

## Evidence and reporting

The workflow and app are prepared in code, but their hosted execution is not established by this repository ZIP alone. Include the real GitHub repository URL, a successful Actions run screenshot, the real Streamlit app URL, and an app screenshot in the final report only after completing those steps.
