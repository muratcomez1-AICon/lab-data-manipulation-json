"""
Titanic Data Analysis and JSON Export
Author: Murat Comez
Description: Analyze Titanic passenger data, engineer features, and export to JSON
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------
# STEP 1: Project setup
# ---------------------------------------------------------------
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "titanic.csv"
JSON_FILE = DATA_DIR / "titanic_processed.json"
DATA_DIR.mkdir(exist_ok=True)

print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")

# ---------------------------------------------------------------
# STEP 2: Import and explore
# ---------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
print(f"\nDataset loaded successfully! Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# ---------------------------------------------------------------
# STEP 3: Descriptive statistics (mean, median, std)
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)

numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_stats = {}
for col in numeric_columns:
    numeric_stats[col] = {
        "mean": round(float(df[col].mean()), 2),
        "median": round(float(df[col].median()), 2),
        "std": round(float(df[col].std()), 2),
    }
    print(f"{col:12s} mean={numeric_stats[col]['mean']:8.2f}  "
          f"median={numeric_stats[col]['median']:8.2f}  "
          f"std={numeric_stats[col]['std']:8.2f}")

# ---------------------------------------------------------------
# STEP 4: Missing values
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("MISSING VALUES ANALYSIS")
print("=" * 50)

missing_data = {}
for col in df.columns:
    missing_count = int(df[col].isna().sum())
    missing_percent = (missing_count / len(df)) * 100
    missing_data[col] = {"count": missing_count, "percent": round(missing_percent, 2)}
    if missing_count > 0:
        print(f"{col:12s} missing: {missing_count:4d} ({missing_percent:.1f}%)")

# ---------------------------------------------------------------
# STEP 5: Feature engineering
# ---------------------------------------------------------------
df_engineered = df.copy()

# Feature 1: FamilySize = siblings/spouses + parents/children + the passenger
df_engineered["FamilySize"] = df_engineered["SibSp"] + df_engineered["Parch"] + 1
print(df_engineered[["SibSp", "Parch", "FamilySize"]].head(10))

# Feature 2: IsAlone = 1 if FamilySize is 1, else 0
df_engineered["IsAlone"] = (df_engineered["FamilySize"] == 1).astype(int)
print(df_engineered[["FamilySize", "IsAlone"]].head(10))


# Feature 3: AgeGroup
def categorize_age(age):
    """Categorize age into groups"""
    if pd.isna(age):
        return "Unknown"
    elif age < 18:
        return "Child"
    elif age < 30:
        return "Young Adult"
    elif age < 50:
        return "Adult"
    else:
        return "Senior"


df_engineered["AgeGroup"] = df_engineered["Age"].apply(categorize_age)
print(df_engineered[["Age", "AgeGroup"]].head(10))

# Extra feature (bonus): Title from the Name column ("Braund, Mr. Owen" -> "Mr")
df_engineered["Title"] = df_engineered["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Analyze: survived vs not survived
print("\n" + "=" * 50)
print("FEATURE ANALYSIS: SURVIVED vs NOT SURVIVED")
print("=" * 50)

print("\nFamily Size by Survival:")
print(df_engineered.groupby("Survived")["FamilySize"].agg(["mean", "median", "std"]))

print("\nSurvival rate by IsAlone:")
print(df_engineered.groupby("IsAlone")["Survived"].mean().round(3))

print("\nSurvival rate by AgeGroup:")
print(df_engineered.groupby("AgeGroup")["Survived"].mean().round(3))

print("\nSurvival rate by Title:")
print(df_engineered.groupby("Title")["Survived"].agg(["mean", "count"]).round(3)
      .sort_values("count", ascending=False).head(6))

survived = df_engineered[df_engineered["Survived"] == 1]
not_survived = df_engineered[df_engineered["Survived"] == 0]
print("\nFamily Size:")
print(f"  Survived mean: {survived['FamilySize'].mean():.2f}")
print(f"  Not Survived mean: {not_survived['FamilySize'].mean():.2f}")
print(f"  Difference: {abs(survived['FamilySize'].mean() - not_survived['FamilySize'].mean()):.2f}")


# ---------------------------------------------------------------
# STEP 6: Classes + JSON export
# ---------------------------------------------------------------
class Passenger:
    """Represents one passenger."""

    def __init__(self, passenger_id, name, age, sex, survived, pclass,
                 fare, embarked=None, family_size=None, is_alone=None,
                 title=None, age_group=None):
        # pd.notna() check -> NaN becomes None (JSON "null")
        self.passenger_id = int(passenger_id) if pd.notna(passenger_id) else None
        self.name = str(name) if pd.notna(name) else None
        self.age = float(age) if pd.notna(age) else None
        self.sex = str(sex) if pd.notna(sex) else None
        self.survived = int(survived) if pd.notna(survived) else None
        self.pclass = int(pclass) if pd.notna(pclass) else None
        self.fare = float(fare) if pd.notna(fare) else None
        self.embarked = str(embarked) if pd.notna(embarked) else None
        self.family_size = int(family_size) if pd.notna(family_size) else None
        self.is_alone = bool(is_alone) if pd.notna(is_alone) else None
        self.title = str(title) if pd.notna(title) else None
        self.age_group = str(age_group) if pd.notna(age_group) else None

    def to_dict(self):
        """Convert passenger to dictionary for JSON serialization."""
        return {
            "passenger_id": self.passenger_id,
            "name": self.name,
            "age": self.age,
            "sex": self.sex,
            "survived": self.survived,
            "pclass": self.pclass,
            "fare": self.fare,
            "embarked": self.embarked,
            "family_size": self.family_size,
            "is_alone": self.is_alone,
            "title": self.title,
            "age_group": self.age_group,
        }


class TitanicDataset:
    """Whole dataset, with statistics and JSON export."""

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.passengers = []
        self._create_passengers()

    def _create_passengers(self):
        """Create Passenger objects from dataframe rows."""
        for idx, row in self.dataframe.iterrows():
            passenger = Passenger(
                passenger_id=row.get("PassengerId", idx),
                name=row.get("Name"),
                age=row.get("Age"),
                sex=row.get("Sex"),
                survived=row.get("Survived"),
                pclass=row.get("Pclass"),
                fare=row.get("Fare"),
                embarked=row.get("Embarked"),
                family_size=row.get("FamilySize"),
                is_alone=row.get("IsAlone"),
                title=row.get("Title"),
                age_group=row.get("AgeGroup"),
            )
            self.passengers.append(passenger)

    def get_summary_stats(self):
        """Summary statistics (None values are skipped in averages)."""
        ages = [p.age for p in self.passengers if p.age is not None]
        fares = [p.fare for p in self.passengers if p.fare is not None]
        survived_count = sum(1 for p in self.passengers if p.survived == 1)
        total = len(self.passengers)
        return {
            "total_passengers": total,
            "survived": survived_count,
            "did_not_survive": total - survived_count,
            "survival_rate": round(survived_count / total, 4),
            "average_age": round(sum(ages) / len(ages), 2),
            "average_fare": round(sum(fares) / len(fares), 2),
        }

    def to_json(self, filename="titanic_processed.json"):
        """Export dataset to JSON file."""
        data = {
            "metadata": {
                "dataset_name": "Titanic Passenger Dataset",
                "export_date": datetime.now().isoformat(),
                "total_passengers": len(self.passengers),
                "survival_rate": round(float(self.dataframe["Survived"].mean()), 4),
                "engineered_features": ["FamilySize", "IsAlone", "AgeGroup", "Title"],
            },
            "statistics": numeric_stats,
            "missing_values": missing_data,
            "summary": self.get_summary_stats(),
            "passengers": [p.to_dict() for p in self.passengers],
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Data exported to {filename}")
        return data


if "df_engineered" in locals() and not df_engineered.empty:
    dataset = TitanicDataset(df_engineered)
    print(f"\nPassengers loaded: {len(dataset.passengers)}")

    print("\nSUMMARY STATS:")
    for key, value in dataset.get_summary_stats().items():
        print(f"  {key}: {value}")

    dataset.to_json(str(JSON_FILE))

# ---------------------------------------------------------------
# STEP 7: Validate the JSON
# ---------------------------------------------------------------
with open(JSON_FILE, "r", encoding="utf-8") as f:
    json_data = json.load(f)

print("\n" + "=" * 50)
print("JSON VALIDATION")
print("=" * 50)
print(f"Top-level keys: {list(json_data.keys())}")
print(f"Passenger records: {len(json_data['passengers'])} (expected 891)")
print(f"Metadata: {json_data['metadata']}")
print(f"Missing values (Age): {json_data['missing_values']['Age']}")
print(f"First passenger: {json_data['passengers'][0]}")
assert len(json_data["passengers"]) == len(df), "Passenger count mismatch!"
print("JSON validation passed!")
