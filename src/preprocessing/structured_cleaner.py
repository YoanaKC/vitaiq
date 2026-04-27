"""
src/preprocessing/structured_cleaner.py
-----------------------------------------
Cleans USDA food data and loads all structured data into SQLite:
- food_items table
- user_profiles table
- biomarkers table (reference ranges)
- query_logs table

Run: python src/preprocessing/structured_cleaner.py
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path

USDA_PATH = Path("data/raw/usda_foods.json")
DB_PATH = Path("data/processed/vitaiq.db")

# Reference ranges for common biomarkers
BIOMARKERS = [
    ("glucose_fasting", "mg/dL", 70, 99, "Fasting blood glucose. Values 100-125 indicate prediabetes."),
    ("hba1c", "%", 4.0, 5.6, "3-month average blood sugar. Below 5.7% is normal."),
    ("ldl_cholesterol", "mg/dL", 0, 99, "LDL cholesterol. Below 100 mg/dL is optimal."),
    ("hdl_cholesterol", "mg/dL", 60, 999, "HDL cholesterol. Above 60 mg/dL is protective."),
    ("triglycerides", "mg/dL", 0, 149, "Blood fats. Below 150 mg/dL is normal."),
    ("vitamin_d", "ng/mL", 30, 100, "Serum 25-hydroxyvitamin D. Below 20 is deficient."),
    ("crp_hs", "mg/L", 0, 1.0, "High-sensitivity C-reactive protein. Below 1.0 is low cardiovascular risk."),
    ("ferritin", "ng/mL", 12, 300, "Iron storage protein. Low values indicate iron deficiency."),
    ("tsh", "mIU/L", 0.4, 4.0, "Thyroid stimulating hormone. Normal range 0.4-4.0."),
    ("bmi", "kg/m2", 18.5, 24.9, "Body Mass Index. 18.5-24.9 is considered normal weight."),
]


def clean_foods(df: pd.DataFrame) -> pd.DataFrame:
    """Clean USDA food records."""
    # Fill missing numeric values with category medians
    numeric_cols = ["calories", "protein_g", "fat_g", "carbs_g", "fiber_g",
                    "sugar_g", "vitamin_c_mg", "vitamin_d_mcg", "calcium_mg",
                    "iron_mg", "magnesium_mg", "omega3_g"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Compute nutritional density score: (protein + fiber) per 100 kcal
    df["nutritional_density_score"] = (
        (df.get("protein_g", 0) + df.get("fiber_g", 0)) /
        (df.get("calories", 1).replace(0, 1)) * 100
    ).round(2)

    df = df.drop_duplicates(subset="fdc_id")
    return df


def build_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── food_items ──────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS food_items (
            fdc_id INTEGER PRIMARY KEY,
            name TEXT,
            data_type TEXT,
            source TEXT,
            calories REAL,
            protein_g REAL,
            fat_g REAL,
            carbs_g REAL,
            fiber_g REAL,
            sugar_g REAL,
            vitamin_c_mg REAL,
            vitamin_d_mcg REAL,
            calcium_mg REAL,
            iron_mg REAL,
            magnesium_mg REAL,
            omega3_g REAL,
            nutritional_density_score REAL
        )
    """)

    # ── user_profiles ────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            age INTEGER,
            sex TEXT,
            weight_kg REAL,
            height_cm REAL,
            health_goals TEXT,      -- JSON array
            conditions TEXT,        -- JSON array
            preferences TEXT,       -- JSON object
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── biomarkers ───────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS biomarkers (
            marker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            unit TEXT,
            ref_low REAL,
            ref_high REAL,
            interpretation TEXT
        )
    """)

    # ── query_logs ───────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            query_id TEXT PRIMARY KEY,
            user_id TEXT,
            query_text TEXT,
            retrieved_doc_ids TEXT,   -- JSON array
            response_text TEXT,
            response_time_ms INTEGER,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # Load food data
    if USDA_PATH.exists():
        with open(USDA_PATH) as f:
            foods_raw = json.load(f)
        df = pd.DataFrame(foods_raw)
        df = clean_foods(df)
        df.to_sql("food_items", conn, if_exists="replace", index=False)
        print(f"Loaded {len(df)} food items into SQLite.")
    else:
        print(f"[WARN] USDA data not found at {USDA_PATH}. Skipping food load.")

    # Load biomarkers
    cur.executemany(
        "INSERT OR IGNORE INTO biomarkers (name, unit, ref_low, ref_high, interpretation) VALUES (?,?,?,?,?)",
        BIOMARKERS,
    )
    conn.commit()
    print(f"Loaded {len(BIOMARKERS)} biomarker reference ranges.")

    conn.close()
    print(f"SQLite database ready: {DB_PATH}")


if __name__ == "__main__":
    build_db()
