"""
Project documentation tool for accessing README, config, and cached reports.
"""

from pathlib import Path
from typing import Any, Dict

from src.genai.tools.base_tool import BaseTool, ToolRegistry


@ToolRegistry.register
class ProjectDocsTool(BaseTool):
    """Access project documentation including README and cached reports."""

    def __init__(self, config: Dict):
        """Initialize with paths to documentation.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.genai_report_path = self.models_dir / "genai_insights_report.md"
        self.readme_path = "README.md"

    @property
    def name(self) -> str:
        return "get_project_docs"

    @property
    def description(self) -> str:
        return """Get project documentation including README and cached GenAI reports.
            Use this to understand the project through written documentation."""

    def execute(self) -> Dict[str, Any]:
        """Retrieve project documentation including README and GenAI report.

        Returns:
            Dictionary with project documentation, or dict with error details
        """
        try:
            docs = {}
            with open(self.readme_path, "r") as f:
                docs["readme"] = f.read()
            try:
                with open(self.genai_report_path, "r") as f:
                    docs["genai_report"] = f.read()
            except FileNotFoundError:
                docs["genai_report"] = "GenAI report not yet generated"
            return docs
        except Exception as e:
            return {"error": f"Failed to load project docs: {e}"}
