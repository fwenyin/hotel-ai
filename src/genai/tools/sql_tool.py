"""SQL query tool for database operations."""

import pandas as pd

from src.data.loader import DataLoader
from src.genai.tools.base_tool import BaseTool, ToolRegistry


@ToolRegistry.register
class SQLTool(BaseTool):
    """Execute SQL queries on the database for data analysis."""

    @property
    def name(self) -> str:
        return "query_database"

    @property
    def description(self) -> str:
        return (
            "Execute SQL query on the hotel bookings SQLite database. "
            "Use only SQLite-compatible syntax (no DATEDIFF, no DATEADD, no TOP — use LIMIT instead). "
            "Table: 'noshow' with columns: booking_id, no_show (0/1), branch, booking_month (int 1-12), "
            "arrival_month (int 1-12), arrival_day (int 1-28), checkout_month (int 1-12), checkout_day (int 1-28), "
            "country, first_time (Yes/No), room, price (text like 'SGD$ 123.45'), platform, "
            "num_adults (text), num_children (float). "
            "To get numeric price use: CAST(REPLACE(REPLACE(price,'SGD$ ',''),'USD$ ','') AS REAL). "
            "Use for aggregations, filtering, and data analysis."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute"
                    " (e.g., 'SELECT country, AVG(no_show)"
                    " FROM noshow GROUP BY country')",
                }
            },
            "required": ["query"],
        }

    def execute(self, query: str = "SELECT * FROM noshow LIMIT 100") -> pd.DataFrame:
        """Execute SQL query on the database.

        Args:
            query: SQL query string. Defaults to sample of bookings.

        Returns:
            DataFrame with query results, or dict with error details
        """
        try:
            db_path = self.config["data"]["database_path"]
            loader = DataLoader(db_path)
            return loader.load_data(query)
        except Exception as e:
            return {"error": f"Failed to execute query: {e}"}
