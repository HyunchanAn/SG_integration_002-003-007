import json

from vsams.paths import DATA_DIR


DB_PATH = DATA_DIR / "database.json"


def load_db():
    if not DB_PATH.exists():
        return []
    with DB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_db(data):
    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def query_recommendation(ai_material, ai_finish):
    db = load_db()
    recommendations = []

    for product in db:
        conditions = product.get("target_condition", {})
        target_materials = conditions.get("material_category", [])
        target_finishes = conditions.get("finish_type", [])

        mat_match = any(m.lower() == ai_material.lower() for m in target_materials)
        fin_match = any(f.lower() == ai_finish.lower() for f in target_finishes)

        if mat_match and fin_match:
            recommendations.append(product)

    if not recommendations and db:
        return [db[0]]

    return recommendations
