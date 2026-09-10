from __future__ import annotations

import json
import base64
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from local_detector import LocalModelError, MODEL_ID, model_status, predict

ROOT = Path(__file__).resolve().parent.parent
INGREDIENT_DB = ROOT / "ingredients.db"
RECIPE_DB = ROOT / "recipes.db"


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    return db


def rows(db: sqlite3.Connection, query: str, params=()):
    return [dict(row) for row in db.execute(query, params).fetchall()]


def products():
    with connect(INGREDIENT_DB) as db:
        return rows(db, """
            SELECT id, name, category, price, unit, zone, storage_type
            FROM products WHERE is_active = 1 ORDER BY zone, name
        """)


def recipes():
    with connect(RECIPE_DB) as db:
        return rows(db, """
            SELECT id, name, description, cook_time_minutes
            FROM recipes ORDER BY name
        """)


def recipe_detail(recipe_id: int):
    """Return a recipe together with its DB-defined ingredients and method."""
    with connect(RECIPE_DB) as db:
        recipe = db.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        if recipe is None:
            return None
        ingredients = rows(db, """
            SELECT * FROM recipe_ingredients
            WHERE recipe_id = ?
        """, (recipe_id,))
        return {"recipe": dict(recipe), "ingredients": ingredients}


def recommendations(names):
    names = sorted({str(name).strip() for name in names if str(name).strip()})
    if not names:
        return []
    placeholders = ",".join("?" for _ in names)
    with connect(RECIPE_DB) as db:
        db.execute("ATTACH DATABASE ? AS ingredient_db", (str(INGREDIENT_DB),))
        query = f"""
            SELECT r.id, r.name, r.description, r.cook_time_minutes,
              COUNT(DISTINCT ri.product_id) AS db_ingredient_count,
              COUNT(DISTINCT CASE WHEN p.name IN ({placeholders}) THEN ri.product_id END) AS matched_count
            FROM recipes r JOIN recipe_ingredients ri ON ri.recipe_id = r.id
            LEFT JOIN ingredient_db.products p
              ON p.id = ri.product_id AND p.name = ri.product_name
            GROUP BY r.id HAVING matched_count > 0
            ORDER BY matched_count DESC, r.name
        """
        result = []
        for item in rows(db, query, names):
            missing = rows(db, f"""
                SELECT p.name FROM recipe_ingredients ri
                JOIN ingredient_db.products p ON p.id = ri.product_id AND p.name = ri.product_name
                WHERE ri.recipe_id = ? AND p.name NOT IN ({placeholders})
                ORDER BY p.name
            """, [item["id"], *names])
            item["match_percent"] = round(item["matched_count"] / max(item["db_ingredient_count"], 1) * 100)
            item["missing_products"] = [x["name"] for x in missing]
            result.append(item)
        return result


def detect_image(data_url: str) -> dict:
    """Run one camera frame through the locally trained YOLO model."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(data_url, validate=True)
    except (ValueError, TypeError) as error:
        return {"configured": False, "model_id": MODEL_ID, "predictions": [],
                "error": f"잘못된 카메라 이미지입니다: {error}"}
    try:
        return {"configured": True, "model_id": MODEL_ID, "predictions": predict(image_bytes)}
    except LocalModelError as error:
        return {"configured": False, "model_id": MODEL_ID, "predictions": [], "error": str(error)}
    except Exception as error:
        return {"configured": True, "model_id": MODEL_ID, "predictions": [], "error": str(error)}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/products":
            return self.send_json(products())
        if path == "/api/recipes":
            return self.send_json(recipes())
        if path.startswith("/api/recipes/"):
            try:
                recipe = recipe_detail(int(path.rsplit("/", 1)[1]))
            except ValueError:
                recipe = None
            return self.send_json(recipe if recipe else {"error": "Recipe not found"}, 200 if recipe else 404)
        if path == "/api/health":
            return self.send_json({"ok": True, "detector": model_status()})
        return self.send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in {"/api/recommendations", "/api/detect"}:
            return self.send_json({"error": "Not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/detect":
                return self.send_json(detect_image(body.get("image", "")))
            return self.send_json(recommendations(body.get("product_names", [])))
        except (ValueError, sqlite3.Error) as error:
            return self.send_json({"error": str(error)}, 400)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Smart Cart API: http://{host}:{port}")
    server.serve_forever()
