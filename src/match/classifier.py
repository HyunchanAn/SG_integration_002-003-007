import os
import sqlite3

class DBFinishClassifier:
    """Classifies surface finish using the 004 module's database."""
    
    def __init__(self, db_path: str = "E:/Github/SG_proj_004/sg_proj_004.db"):
        self.db_path = db_path

    def predict_label(self, roughness: float, gloss: float) -> str:
        """Queries the adherend_properties table in sg_proj_004.db to find the nearest match."""
        if not os.path.exists(self.db_path):
            return "Unknown (DB Not Found)"
        
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                SELECT product_name, roughness_md, gloss_md 
                FROM adherend_properties
            """)
            rows = c.fetchall()
            conn.close()
            
            best_label = "Unknown"
            min_distance = float("inf")
            
            for name, r_md, g_md in rows:
                if r_md is None or g_md is None:
                    continue
                # Normalize distances: roughness error scaled by 0.1, gloss error by 100.0
                dist = (abs(roughness - r_md) / 0.1) + (abs(gloss - g_md) / 100.0)
                if dist < min_distance:
                    min_distance = dist
                    best_label = name
            
            # Format nicely
            if best_label == "BA":
                return "BA (Bright Annealed)"
            elif best_label == "HL":
                return "HL (Hairline)"
            elif best_label == "SM":
                return "SM (Super Mirror)"
            elif best_label == "#4":
                return "#4 (Rough)"
            elif best_label == "2B":
                return "2B (2B/2D)"
            else:
                return best_label
        except Exception as e:
            return f"Error: {e}"
