"""Data Science Agent for autonomous analysis tasks."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from src.genai.tools import model_metadata_tool  # noqa: F401
from src.genai.tools import project_docs_tool  # noqa: F401
from src.genai.tools import sql_tool  # noqa: F401
from src.genai.tools.base_tool import ToolRegistry
from src.utils.clients import DEFAULT_MODEL, get_genai_client

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""

    tool: str
    output: Any
    reasoning: str


class DataScienceAgent:
    """AI Agent for autonomous data science tasks."""

    def __init__(
        self,
        config: Dict[str, Any],
        max_iterations: int = 5,
        temperature: float = 0.0,
    ) -> None:
        """Initialize the DataScienceAgent.

        Args:
            config: Configuration dictionary containing genai settings.
            max_iterations: Maximum number of reasoning-action iterations.
        """
        self.max_iterations = max_iterations
        self.config = config
        self.action_history: List[AgentAction] = []
        self.model_name = DEFAULT_MODEL
        self.temperature = temperature
        self.llm = get_genai_client()
        self._init_tools()

    def _init_tools(self) -> None:
        self.tools = ToolRegistry.instantiate_all(self.config)
        logger.info(f"Loaded {len(self.tools)} tools: {list(self.tools.keys())}")

    def generate_response(
        self,
        prompt: str,
    ) -> str:
        """Call Azure OpenAI API.

        Args:
            prompt: User prompt text

        Returns:
            LLM response text
        """
        try:
            system_message = """
                You are an expert data scientist analyzing hotel no-show prediction models.
                Users will be asking you questions related to these models and dataset.
                Ensure that any technical terms are clearly and concisely explained.
            """
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            response = self.llm.chat.completions.create(
                messages=messages, temperature=self.temperature, model=self.model_name
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Azure OpenAI API: {str(e)}"

    def _reason(self, task: str, observations: List[str]) -> Dict[str, Any]:
        """Reasoning step using LLM to decide next action.

        Args:
            task: Original task description
            observations: List of observations from previous actions

        Returns:
            Dictionary with 'action', 'tool', 'reasoning', and 'input'
        """
        tool_descriptions = []
        for tool in self.tools.values():
            tool_info = f"- {tool.name}: {tool.description}"
            if tool.input_schema:
                tool_info += (
                    f"\n  Parameters: {json.dumps(tool.input_schema, indent=2)}"
                )
            tool_descriptions.append(tool_info)

        tool_catalog = "Available Tools:\n" + "\n".join(tool_descriptions)
        observations_text = "\n".join(
            [f"{i+1}. {obs}" for i, obs in enumerate(observations)]
        )

        prompt = f"""Analyze the task and observations, then decide the next action.

        Task: {task}

        {tool_catalog}

        Previous Observations:
        {observations_text if observations else "No observations yet"}

        Based on the task and observations, what should be the next action?

        Respond strictly in JSON format:
        {{
            "action": "use_tool" or "finish",
            "tool": "tool_name" (if action is use_tool),
            "tool_input": {{}} (parameters for the tool, matching its input_schema),
            "reasoning": "explanation of why this action"
        }}

        If you have enough information to answer the task, use "finish". Otherwise, select the most appropriate tool."""

        response = self.generate_response(prompt)
        return json.loads(response)

    def _observe(self, tool_name: str, tool_output: Any) -> str:
        """Generate observation from tool output.

        Args:
            tool_name: Name of tool that was executed
            tool_output: Output from the tool

        Returns:
            Natural language observation with actual data (truncated if needed)
        """
        if isinstance(tool_output, pd.DataFrame):
            rows, cols = tool_output.shape
            output_str = f"DataFrame with {rows} rows and {cols} columns\n"
            output_str += f"Columns: {list(tool_output.columns)}\n\n"
            output_str += "First 50 rows:\n"
            output_str += tool_output.head(50).to_string()
            return f"Tool '{tool_name}' returned:\n{output_str}"

        # Try JSON serialization for dicts/lists
        try:
            output_str = json.dumps(tool_output, default=str, indent=2)
        except (TypeError, ValueError):
            output_str = str(tool_output)
        return f"Tool '{tool_name}' returned:\n{output_str}"

    def _generate_final_summary(
        self,
        task: str,
        observations: List[str],
    ) -> str:
        """Generate comprehensive final summary using LLM.

        Args:
            task: Original task description.
            observations: List of observation strings collected during execution.

        Returns:
            Natural language summary with insights and recommendations.
        """
        context_parts = [f"Task: {task}\n"]
        context_parts.append(f"Actions taken: {len(observations)}\n")
        context_parts.append("\nObservations:")
        for i, obs in enumerate(observations, 1):
            context_parts.append(f"{i}. {obs}")

        context = "\n".join(context_parts)

        prompt = f"""Based on the following agent execution, provide a comprehensive summary:

        {context}

        Provide your response in well-formatted markdown with the following structure:

        ### Key Findings from the Analysis
        - List specific findings with actual numbers and data points from the results
        - Include feature names, metrics, or values that answer the question

        ### Business Insights and Interpretation
        - Explain what these findings mean for hotel operations
        - Connect the data to business outcomes

        ### Actionable Recommendations for Hotel Management
        1. Numbered list of specific actions hotel management can take
        2. Each recommendation should be practical and implementable

        Use proper markdown spacing (blank lines between sections). Be concise but specific."""

        try:
            summary = self.generate_response(prompt)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Completed {len(observations)} actions. Check logs for details."

    def execute_task(self, task: str) -> str:
        """Execute a task using the ReAct (Reasoning + Acting) pattern.

        The agent will alternate between reasoning (deciding next action via
        the LLM) and acting (invoking registered tools) until the task is
        completed or max_iterations is reached.

        Args:
            task: Natural language description of the task to perform.

        Returns:
            Natural language summary of findings and recommendations.
        """
        logger.info(f"Agent Task: {task}")

        observations: List[str] = []

        for iteration in range(self.max_iterations):
            logger.info(f"[Iteration {iteration + 1}/{self.max_iterations}]")

            try:
                decision = self._reason(task, observations)
                logger.info(f"Reasoning: {decision['reasoning']}")
            except Exception as e:
                logger.error(f"Reasoning error: {e}")
                break

            if decision["action"] == "finish":
                logger.info("Task complete")
                break

            tool_name = decision["tool"]
            tool_input = decision.get("tool_input", {})
            logger.info(f"Action: {tool_name} with input: {tool_input}")

            try:
                tool = self.tools[tool_name]
                tool_output = tool.execute(**tool_input)
                action = AgentAction(
                    tool=tool_name, output=tool_output, reasoning=decision["reasoning"]
                )
                self.action_history.append(action)
                observation = self._observe(tool_name, tool_output)
                observations.append(observation)
                logger.info(f"Observation: {observation}")
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                logger.error(error_msg)
                observations.append(error_msg)

        summary = self._generate_final_summary(task, observations)
        return summary
