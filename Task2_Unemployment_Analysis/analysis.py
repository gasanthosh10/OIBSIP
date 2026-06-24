"""Task 2: Clean, analyse, and visualise unemployment data in India."""

from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

RATE = "Unemployment Rate (%)"
EMPLOYED = "Estimated Employed"
PARTICIPATION = "Labour Participation Rate (%)"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both datasets and standardise their column names."""
    historical = pd.read_csv(DATA_DIR / "Unemployment in India.csv")
    recent = pd.read_csv(DATA_DIR / "Unemployment_Rate_upto_11_2020.csv")

    for frame in (historical, recent):
        frame.columns = frame.columns.str.strip()
        frame.rename(
            columns={
                "Estimated Unemployment Rate (%)": RATE,
                "Estimated Labour Participation Rate (%)": PARTICIPATION,
            },
            inplace=True,
        )
        frame.dropna(how="all", inplace=True)
        frame["Date"] = pd.to_datetime(frame["Date"].str.strip(), dayfirst=True)
        frame["Region"] = frame["Region"].str.strip()

    recent.rename(columns={"Region.1": "Zone"}, inplace=True)
    return historical, recent


def generate_report() -> dict[str, float | str]:
    """Generate charts, cleaned exports, and headline statistics."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    historical, recent = load_data()
    historical.to_csv(OUTPUT_DIR / "cleaned_unemployment_india.csv", index=False)
    recent.to_csv(OUTPUT_DIR / "cleaned_unemployment_2020.csv", index=False)

    monthly = (
        historical.groupby("Date", as_index=False)
        .agg({RATE: "mean", EMPLOYED: "sum", PARTICIPATION: "mean"})
        .sort_values("Date")
    )
    monthly.to_csv(OUTPUT_DIR / "monthly_national_summary.csv", index=False)

    pre_covid = historical[historical["Date"] < "2020-03-01"][RATE].mean()
    lockdown = historical[
        historical["Date"].between("2020-03-01", "2020-05-31")
    ][RATE].mean()
    peak_row = monthly.loc[monthly[RATE].idxmax()]
    state_avg = recent.groupby("Region")[RATE].mean().sort_values(ascending=False)

    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid", palette="crest")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly["Date"], monthly[RATE], marker="o", linewidth=2.5)
        ax.axvspan(
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2020-05-31"),
            color="#ff6b6b",
            alpha=0.15,
            label="First lockdown period",
        )
        ax.set(title="Average Unemployment Rate in India", xlabel="", ylabel=RATE)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "national_unemployment_trend.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        state_avg.sort_values().plot(kind="barh", ax=ax, color="#27ae60")
        ax.set(
            title="Average Unemployment Rate by State (2020)",
            xlabel=RATE,
            ylabel="",
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "state_comparison.png", dpi=180)
        plt.close(fig)

    summary = {
        "records_after_cleaning": int(len(historical)),
        "pre_covid_average_rate": round(float(pre_covid), 2),
        "lockdown_average_rate": round(float(lockdown), 2),
        "peak_month": peak_row["Date"].strftime("%B %Y"),
        "peak_rate": round(float(peak_row[RATE]), 2),
        "highest_average_state_2020": str(state_avg.index[0]),
        "highest_average_state_rate": round(float(state_avg.iloc[0]), 2),
    }
    pd.Series(summary, name="value").to_csv(OUTPUT_DIR / "key_findings.csv")
    return summary


if __name__ == "__main__":
    findings = generate_report()
    print("Task 2 analysis completed.")
    if plt is None:
        print("Static charts skipped; install requirements.txt to generate them.")
    for key, value in findings.items():
        print(f"{key}: {value}")
