from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from vsams.paths import DATA_DIR


class SubstrateDB:
    """Database handler for substrate physical properties and visual embeddings.

    Loads a physical property registry from Excel and comparison visual embeddings
    from a PyTorch archive, enabling nearest-neighbor lookup and cross-modal search.
    """

    def __init__(
        self,
        excel_path: Optional[Union[str, Path]] = None,
        visual_library_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the SubstrateDB with target registry paths.

        Args:
            excel_path: Path to the Excel property matrix. Defaults to target directory location.
            visual_library_path: Path to the serialized visual tensor archive. Defaults to target data folder.
        """
        if excel_path is None:
            excel_path = DATA_DIR / "substrate_properties.xlsx"
        if visual_library_path is None:
            visual_library_path = DATA_DIR / "visual_library.pth"

        self.excel_path = Path(excel_path).resolve()
        self.df: Optional[pd.DataFrame] = None
        self.visual_library: Optional[List[Dict[str, Any]]] = None
        self.load_db()
        self.load_visual_library(visual_library_path)

    def load_db(self) -> None:
        """Loads the physical steel finish database from the Excel spreadsheet.

        Parses roughness, gloss, and surface energy metrics across rolling dimensions (MD/TD).
        """
        if not self.excel_path.exists():
            print(f"Warning: Database file not found at {self.excel_path}")
            return

        try:
            # We use header=None and manually slice to avoid issues with multi-index headers.
            df_raw = pd.read_excel(self.excel_path, header=None)

            data = df_raw.iloc[6:].copy()
            mapping = {
                1: "no",
                2: "classification",
                4: "product_name",
                5: "roughness_md",
                6: "roughness_td",
                7: "gloss_md",
                8: "gloss_td",
                9: "energy_md",
                10: "energy_td",
            }

            self.df = data[list(mapping.keys())].rename(columns=mapping)
            self.df = self.df.dropna(subset=["product_name"]).reset_index(drop=True)

            numeric_cols = ["roughness_md", "roughness_td", "gloss_md", "gloss_td"]
            for col in numeric_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            self.df["roughness_avg"] = self.df[["roughness_md", "roughness_td"]].mean(
                axis=1
            )
            self.df["gloss_avg"] = self.df[["gloss_md", "gloss_td"]].mean(axis=1)
            self.df = self.df.dropna(subset=["roughness_avg", "gloss_avg"]).reset_index(
                drop=True
            )

            print(f"Successfully loaded {len(self.df)} products from DB.")
        except Exception as e:
            print(f"Error loading Excel DB: {e}")

    def load_visual_library(self, library_path: Union[str, Path]) -> None:
        """Loads the pre-calculated PyTorch visual reference embedding archive.

        Args:
            library_path: Target path to the visual reference serialization.
        """
        library_path = Path(library_path)
        if not library_path.exists():
            self.visual_library = None
            return

        import torch

        try:
            # weights_only=False is required because the library contains numpy arrays.
            self.visual_library = torch.load(library_path, weights_only=False)
            count = len(self.visual_library) if self.visual_library is not None else 0
            print(f"Loaded visual library with {count} products.")
        except Exception as e:
            print(f"Error loading visual library: {e}")
            self.visual_library = None

    def find_closest(
        self, roughness: float, glossiness: float
    ) -> Optional[Dict[str, Any]]:
        """Finds the closest single product matching the target roughness and gloss metrics.

        Args:
            roughness: Estimated physical roughness (Ra).
            glossiness: Estimated gloss reflectivity (%).

        Returns:
            Dictionary containing the matching product attributes, or None if DB is empty.
        """
        if self.df is None or self.df.empty:
            return None

        r_err = (self.df["roughness_avg"] - roughness) / 0.1
        g_err = (self.df["gloss_avg"] - glossiness) / 10.0
        self.df["distance"] = np.sqrt(r_err**2 + g_err**2)

        closest_idx = self.df["distance"].idxmin()
        closest = self.df.loc[closest_idx]
        return closest.to_dict()

    def find_closest_top_k(
        self, roughness: float, glossiness: float, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Finds the top-K closest matching steel products in physical property space.

        Args:
            roughness: Target physical roughness.
            glossiness: Target contrast-based gloss reflectivity.
            k: Maximum number of closest matches to return.

        Returns:
            A list of dictionary mappings representing top matches.
        """
        if self.df is None or self.df.empty:
            return []

        r_err = (self.df["roughness_avg"] - roughness) / 0.1
        g_err = (self.df["gloss_avg"] - glossiness) / 10.0
        self.df["distance"] = np.sqrt(r_err**2 + g_err**2)
        top_k = self.df.nsmallest(k, "distance")

        results = []
        for _, row in top_k.iterrows():
            results.append(
                {
                    "product_name": row["product_name"],
                    "distance": float(row["distance"]),
                    "roughness": float(row["roughness_avg"]),
                    "gloss": float(row["gloss_avg"]),
                }
            )
        return results

    def find_visual_match(
        self, input_features: np.ndarray, k: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Performs latent visual vector cosine similarity matching against reference library.

        Args:
            input_features: Extracted 2048-dim latent space representation.
            k: Top-K matching items to return.

        Returns:
            Sorted visual match candidates with similarity score metrics.
        """
        if self.visual_library is None:
            return None

        matches = []
        feat_norm = np.linalg.norm(input_features)

        for item in self.visual_library:
            ref_features = item["features"]
            dot = np.dot(ref_features, input_features)
            ref_norms = np.linalg.norm(ref_features, axis=1)
            similarities = dot / (ref_norms * feat_norm + 1e-8)

            score = float(np.mean(similarities))
            matches.append(
                {
                    "product_name": item["product_name"],
                    "similarity": score,
                    "ref_image": item["ref_image"],
                }
            )

        matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        return matches[:k]


if __name__ == "__main__":
    db = SubstrateDB()
    if db.df is not None:
        print(db.df[["product_name", "roughness_avg", "gloss_avg"]].head())
        res = db.find_closest(0.4, 20.0)
        print("\nClosest Match for (0.4, 20.0):")
        if res:
            print(
                f"Name: {res['product_name']}, "
                f"Ra: {res['roughness_avg']:.4f}, "
                f"Gloss: {res['gloss_avg']:.2f}"
            )
