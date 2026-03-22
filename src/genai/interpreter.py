"""GenAI interpreter for ML results interpretation."""

from typing import Dict

from src.utils.clients import DEFAULT_MODEL, get_genai_client


class GenAIInterpreter:
    """Use Azure OpenAI models to interpret ML model outputs."""

    def __init__(
        self,
        temperature: float = 0.0,
    ):
        """Initialize GenAI interpreter with Azure OpenAI client.

        Args:
            temperature: Sampling temperature
        """
        self.model_name = DEFAULT_MODEL
        self.temperature = temperature
        self.llm = get_genai_client()

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
                You will present your findings to business leaders, ensure that any technical terms
                are clearly and concisely explained.
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

    def interpret_feature_importance(
        self, feature_importance: Dict[str, float], top_n: int = 15
    ) -> str:
        """Interpret feature importance in business terms.

        Args:
            feature_importance: Dictionary of feature names to importance scores
            top_n: Number of top features to interpret

        Returns:
            Natural language interpretation of feature importance
        """
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_n]

        feature_text = "\n".join(
            [f"- {feat}: {score:.3f}" for feat, score in sorted_features]
        )

        prompt = f"""These are the top feature importance scores from a hotel no-show prediction model:

        {feature_text}

        Provide:
        1. A short bullet list of the top 5 business drivers of no-shows (1 sentence each, plain language, no scores)
        2. 3-4 actionable bullet points for hotel management based on these drivers

        Be concise. No technical jargon. No metric definitions. This is for business stakeholders."""

        return self.generate_response(prompt)

    def interpret_model_performance(
        self, metrics: Dict[str, float], model_name: str
    ) -> str:
        """Interpret model performance metrics.

        Args:
            metrics: Dictionary of metric names to values
            model_name: Name of the model

        Returns:
            Natural language interpretation of model performance
        """
        metrics_text = "\n".join(
            [f"- {metric}: {value:.3f}" for metric, value in metrics.items()]
        )

        prompt = f"""Analyze the performance of this {model_name} for hotel no-show prediction:

        {metrics_text}

        Provide:
        1. Overall model quality assessment (is this good enough for production?)
        2. Which metrics indicate strengths/weaknesses (compare precision vs recall trade-offs)
        3. Business implications of this performance (cost of errors, revenue impact)
        4. Recommendations for improvement

        Keep response concise, practical, and focused on hotel operations."""

        return self.generate_response(prompt)

    def explain_prediction(
        self, prediction: int, probability: float, feature_values: Dict[str, any]
    ) -> str:
        """Explain a single prediction in business terms.

        Args:
            prediction: Predicted class (0 or 1)
            probability: Prediction probability
            feature_values: Dictionary of feature names to values

        Returns:
            Natural language explanation of prediction
        """
        outcome = "No-Show" if prediction == 1 else "Show"

        features_text = "\n".join(
            [f"- {feat}: {val}" for feat, val in list(feature_values.items())[:10]]
        )

        prompt = f"""This guest has booked a room and the model predicts if they will show up.
        Explain this hotel booking prediction:

        Prediction: {outcome}
        No-Show Probability: {probability:.1%}
        Show Probability: {1 - probability:.1%}

        Booking details:
        {features_text}

        Provide:
        1. Non-technical explanation of the prediction (the no-show probability means the chance
           the guest will NOT show up, and the show probability is the chance they WILL show up)
        2. Main factors influencing this prediction
        3. Recommended action for hotel staff (should they follow up, prepare for potential no-show, etc.)

        Keep response practical and actionable for hotel staff."""

        return self.generate_response(prompt)
