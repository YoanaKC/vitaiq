"""
src/ingestion/usda_fetcher.py
------------------------------
Fetches nutritional data from the USDA FoodData Central REST API
and saves it to data/raw/usda_foods.json.

Free API key at: https://fdc.nal.usda.gov/api-guide.html
Set USDA_API_KEY in your .env file (or the script uses DEMO_KEY for testing).

Run: python src/ingestion/usda_fetcher.py
"""

import json
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
OUTPUT_PATH = Path("data/raw/usda_foods.json")

# Food categories to pull — focused on longevity/wellness
SEARCH_QUERIES = [
    "salmon", "sardines", "blueberries", "broccoli", "spinach",
    "avocado", "olive oil", "almonds", "walnuts", "lentils",
    "eggs", "Greek yogurt", "turmeric", "ginger", "garlic",
    "sweet potato", "quinoa", "dark chocolate", "green tea",
    "mushrooms", "flaxseed", "chia seeds", "kale", "beets",
]

MAX_PER_QUERY = 5
NUTRIENT_IDS = {
    "calories": 1008,
    "protein_g": 1003,
    "fat_g": 1004,
    "carbs_g": 1005,
    "fiber_g": 1079,
    "sugar_g": 2000,
    "vitamin_c_mg": 1162,
    "vitamin_d_mcg": 1114,
    "calcium_mg": 1087,
    "iron_mg": 1089,
    "magnesium_mg": 1090,
    "omega3_g": 1404,
}


def search_foods(query: str, page_size: int = 5) -> list[dict]:
    """Search FoodData Central for a food term."""
    url = f"{BASE_URL}/foods/search"
    params = {
        "query": query,
        "pageSize": page_size,
        "api_key": API_KEY,
        "dataType": "Foundation,SR Legacy",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("foods", [])


def extract_nutrients(food: dict) -> dict:
    """Extract key nutrient values from a food record."""
    nutrients = {n["nutrientId"]: n.get("value", 0) for n in food.get("foodNutrients", [])}
    result = {}
    for name, nid in NUTRIENT_IDS.items():
        result[name] = nutrients.get(nid, None)
    return result


def run():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_foods = []
    seen_ids = set()

    for query in SEARCH_QUERIES:
        print(f"Fetching foods for: '{query}' ...")
        try:
            foods = search_foods(query, MAX_PER_QUERY)
            for food in foods:
                fdc_id = food.get("fdcId")
                if fdc_id in seen_ids:
                    continue
                seen_ids.add(fdc_id)

                nutrients = extract_nutrients(food)
                all_foods.append({
                    "fdc_id": fdc_id,
                    "name": food.get("description", ""),
                    "data_type": food.get("dataType", ""),
                    "source": "USDA FoodData Central",
                    **nutrients,
                })

            print(f"  Total foods so far: {len(all_foods)}")
            time.sleep(0.5)

        except Exception as e:
            print(f"  [ERROR] Failed for '{query}': {e}")
            time.sleep(3)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_foods, f, indent=2)

    print(f"\nDone. {len(all_foods)} food items saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
