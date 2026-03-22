"""Model metadata tool for accessing trained model information."""

import json
from pathlib import Path
from typing import Any, Dict

from src.genai.tools.base_tool import BaseTool, ToolRegistry


@ToolRegistry.register
class ModelMetadataTool(BaseTool):
    """Access trained model metadata, performance metrics, and feature list and importance."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.results_path = self.models_dir / "results.json"

    @property
    def name(self) -> str:
        return "get_model_metadata"

    @property
    def description(self) -> str:
        return """Get metadata about trained models including performance metrics
            feature list and importance, hyperparameters, and champion model information.
            Use this to answer questions about model performance, which features
            are important, or which model is currently selected."""

    def execute(self) -> Dict[str, Any]:
        """Load model metadata from results file.

        Returns:
            Dictionary with model results, or dict with error details
        """
        try:
            with open(self.results_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "error": "No model results found. Run 'python ml_pipeline.py' to train models first."
            }
        except Exception as e:
            return {"error": f"Failed to load model metadata: {e}"}
