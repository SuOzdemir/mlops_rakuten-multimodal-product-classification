"""Loadable MLflow model for a complete Rakuten serving bundle."""

from __future__ import annotations

import base64
import io

import mlflow.pyfunc
from PIL import Image


class RakutenMultimodalModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around the production late-fusion predictor.

    Input is a pandas DataFrame with ``designation`` and ``description`` plus
    an optional ``image_path`` or base64-encoded ``image_base64`` column.
    The output is one JSON-compatible prediction dictionary per row.
    """

    def load_context(self, context):
        from streamlit_app.services.raw_late_fusion_predictor import load_assets

        self.assets = load_assets(context.artifacts["serving_assets"])

    @staticmethod
    def _image_from_row(row):
        image_path = row.get("image_path")
        if image_path:
            return Image.open(str(image_path)).convert("RGB")
        image_base64 = row.get("image_base64")
        if image_base64:
            return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        return None

    def predict(self, context, model_input, params=None):
        from streamlit_app.services.raw_late_fusion_predictor import predict

        image_weight = float((params or {}).get("image_weight", 0.45))
        return [
            predict(
                assets=self.assets,
                designation=row.get("designation", ""),
                description=row.get("description", ""),
                image=self._image_from_row(row),
                image_weight=image_weight,
            )
            for _, row in model_input.iterrows()
        ]
