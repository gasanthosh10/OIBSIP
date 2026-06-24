# OIBSIP — Data Science Internship

Projects completed for the Oasis Infobyte Data Science Internship.

## Task 2 — Unemployment Analysis with Python

This project analyses unemployment in India across states, regions, rural and
urban areas, and time. It includes a Python analysis pipeline and an interactive
Streamlit dashboard.

### Key findings

- 740 valid historical records remained after removing 28 blank rows.
- Average unemployment before March 2020 was **9.51%**.
- Average unemployment during March–May 2020 increased to **19.68%**.
- The monthly unemployment rate peaked at **24.88% in May 2020**.
- Haryana had the highest average state unemployment rate in the supplied 2020
  dataset at **27.48%**.

### Features

- Data cleaning and date standardisation
- National unemployment trend analysis
- COVID-19 lockdown-period comparison
- Rural and urban comparison
- State and regional rankings
- Interactive filters and downloadable results

### Run locally

```bash
git clone https://github.com/gasanthosh10/OIBSIP.git
cd OIBSIP
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python Task2_Unemployment_Analysis/analysis.py
streamlit run Task2_Unemployment_Analysis/app.py
```

## Repository progress

- [x] Task 2 — Unemployment Analysis with Python
- [x] Task 4 — Email Spam Detection with Machine Learning
- [ ] Task 5 — Sales Prediction Using Python

## Task 4 — Email Spam Detection

This project uses TF-IDF text features and class-balanced logistic regression
to classify messages as legitimate or spam.

### Verified performance

- Test accuracy: **97.68%**
- Spam precision: **92.80%**
- Spam recall: **88.55%**

Run the interactive detector with:

```bash
streamlit run Task4_Email_Spam_Detection/app.py
```

## Author

**G.A.Santhosh**
