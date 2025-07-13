from judgeval.tracer import Tracer
from judgeval.scorers import AnswerRelevancyScorer, ToolOrderScorer
from judgeval import JudgmentClient

# Initialize the tracer with your project name
judgment = Tracer(project_name="movie-recommendation-bot")

# Modified tool call to run an evaluation for a sample LLM call
@judgment.observe(span_type="tool")
def my_tool():
    query = "What is the capital of France?"

    # Sample result retrieved from your LLM call
    sample_llm_result = "Paris"

    # Run an evaluation with the Judgeval AnswerRelevancyScorer
    judgment.async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=query,
        actual_output=sample_llm_result,
        model="gpt-4o",
    )
    return sample_llm_result

# Use the @judgment.observe decorator to trace the function
@judgment.observe(span_type="function")
def sample_function():
    tool_called = my_tool()
    message = "Called my_tool() and got: " + tool_called
    return message

if __name__ == "__main__":
    res = sample_function()
    print(res)
