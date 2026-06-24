# Submission Guide

## 1. Publish the repository

Create a public GitHub repository named **OIBSIP**, then run these commands from
the project directory:

```bash
git init
git add .
git commit -m "Complete Oasis Infobyte data science internship tasks"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/OIBSIP.git
git push -u origin main
```

Replace the placeholder repository URL in the main `README.md`.

## 2. Create live demos

Deploy each `app.py` through Streamlit Community Cloud. For every deployment:

1. Connect the `OIBSIP` GitHub repository.
2. Select the appropriate `app.py` path.
3. Use `requirements.txt` from the repository root.
4. Copy the resulting public link into the submission form and README.

App paths:

- `Task2_Unemployment_Analysis/app.py`
- `Task4_Email_Spam_Detection/app.py`
- `Task5_Sales_Prediction/app.py`

## 3. Demonstration outline

### Task 2 — Unemployment analysis

- Explain the two datasets and cleaning of 28 blank rows.
- Show the national trend and COVID-19 lockdown highlight.
- Compare rural/urban areas and filter by state.
- Mention the May 2020 peak of 24.88%.

### Task 4 — Spam detection

- Explain TF-IDF and logistic regression in simple language.
- Show the 97.68% test accuracy and confusion matrix.
- Test one spam message and one legitimate message.
- Mention class balancing and the educational-use limitation.

### Task 5 — Sales prediction

- Explain TV, radio, and newspaper as model inputs.
- Compare linear regression with random forest.
- Show the selected model's 0.980 R² and 0.661 MAE.
- Change the budgets and demonstrate the forecast.

## 4. Final checklist

- Repository name is `OIBSIP`.
- Repository is public and has a complete README.
- All three tasks are committed.
- Live demo links work in an incognito browser.
- Demonstration video has clear voice-over and visible results.
- Submission uses the official form.
- Files follow the `YourName_TaskNumber` naming format.

