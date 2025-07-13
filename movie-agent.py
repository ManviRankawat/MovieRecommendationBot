import os
from typing import TypedDict, Sequence, Dict, Any, Optional, List, Union
from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from judgeval.common.tracer import Tracer
from judgeval.integrations.langgraph import JudgevalCallbackHandler
from judgeval.scorers import AnswerRelevancyScorer, ToolOrderScorer
from judgeval import JudgmentClient
from judgeval.data import Example

# Load environment variables
load_dotenv()

client = JudgmentClient()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
chat_model = ChatOpenAI(model="gpt-4", temperature=0)

judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    project_name="movie-recommendation-bot",
    enable_monitoring=True,  # Explicitly enable monitoring
    deep_tracing=False # Disable deep tracing when using LangGraph handler
)

# Define the state type
class State(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]
    preferences: Dict[str, str]
    search_results: Dict[str, Any]
    recommendations: str
    current_question_idx: int
    questions: Sequence[str]

# Node functions
def initialize_state() -> State:
    """Initialize the state with questions and predefined answers."""
    questions = [
        "What are some of your favorite movie genres?",
        "Who are some of your favorite actors or actresses?",
        "Do you have any favorite directors?",
        "What are some movies you've enjoyed recently?",
        "Do you prefer newer releases or classic films?",
        "Are you looking for any specific mood or theme in movies?"
    ]
    
    # Predefined answers for testing
    answers = [
        "Action, Sci-Fi, and Thriller",
        "Leonardo DiCaprio, Margot Robbie, and Ryan Gosling", 
        "Christopher Nolan, Denis Villeneuve, and Greta Gerwig",
        "Inception, Barbie, and Blade Runner 2049",
        "I enjoy both new and classic films",
        "Thought-provoking and visually stunning movies"
    ]
    
    # Initialize messages with questions and answers alternating
    messages = []
    for question, answer in zip(questions, answers):
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    
    return {
        "messages": messages,
        "preferences": {},
        "search_results": {},
        "recommendations": "",
        "current_question_idx": 0,
        "questions": questions
    }

def ask_question(state: State) -> State:
    """Process the next question-answer pair."""
    if state["current_question_idx"] >= len(state["questions"]):
        return state
    
    # The question is already in messages, just return the state
    return state

def process_answer(state: State) -> State:
    """Process the predefined answer and store it in preferences."""
    messages = state["messages"]
    
    # Ensure we have both a question and an answer
    if len(messages) < 2 or state["current_question_idx"] >= len(state["questions"]):
        return state
    
    try:
        last_question = state["questions"][state["current_question_idx"]]
        # Get the answer from messages - it will be after the question
        answer_idx = (state["current_question_idx"] * 2) + 1  # Calculate the index of the answer
        last_answer = messages[answer_idx].content
        
        state["preferences"][last_question] = last_answer
        state["current_question_idx"] += 1
        
        # Print the Q&A for visibility
        print(f"\nQ: {last_question}")
        print(f"A: {last_answer}\n")
        
    except IndexError:
        return state
    
    return state

def search_movie_info(state: State) -> State:
    """Search for movie recommendations based on preferences."""
    preferences = state["preferences"]
    search_results = {}
    
    # Search for genre-based recommendations
    if preferences.get("What are some of your favorite movie genres?"):
        genre_query = f"Best {preferences['What are some of your favorite movie genres?']} movies 2024 2023"
        search_results["genre_based"] = tavily_client.search(
            query=genre_query,
            search_depth="advanced",
            max_results=5
        )
    
    # Search for actor-based recommendations
    if preferences.get("Who are some of your favorite actors or actresses?"):
        actor_query = f"Best movies with {preferences['Who are some of your favorite actors or actresses?']} recent films"
        search_results["actor_based"] = tavily_client.search(
            query=actor_query,
            search_depth="advanced",
            max_results=5
        )
    
    # Search for director-based recommendations
    if preferences.get("Do you have any favorite directors?"):
        director_query = f"Movies by {preferences['Do you have any favorite directors?']} filmography"
        search_results["director_based"] = tavily_client.search(
            query=director_query,
            search_depth="advanced",
            max_results=5
        )
    
    # Search for mood-based recommendations
    mood_question = "Are you looking for any specific mood or theme in movies?"
    if preferences.get(mood_question):
        mood_query = f"{preferences[mood_question]} movie recommendations 2024"
        search_results["mood_based"] = tavily_client.search(
            query=mood_query,
            search_depth="advanced",
            max_results=5
        )
    
    state["search_results"] = search_results
    return state

def generate_recommendations(state: State) -> State:
    """Generate personalized movie recommendations using ChatOpenAI."""
    preferences = state["preferences"]
    search_results = state["search_results"]
    
    # Prepare context from search results
    context = ""
    for category, results in search_results.items():
        if results and results.get("results"):
            context += f"\n{category.replace('_', ' ').title()} Search Results:\n"
            for result in results.get("results", []):
                content_preview = result.get('content', '')[:200]
                context += f"- {result.get('title')}: {content_preview}...\n"
        else:
            context += f"\nNo search results found for {category.replace('_', ' ').title()}\n"
    
    # Create messages for the Chat Model
    system_message = SystemMessage(content="""
    You are a movie recommendation expert. Based on the user's preferences for genres, actors, directors, and mood, 
    suggest personalized movie recommendations. Consider their stated preferences and provide diverse options 
    that match their interests. Focus on both popular and hidden gem movies that align with their tastes.
    """)

    user_prompt = f"""
    Based on the user's movie preferences, suggest 5-7 movies. For each movie, include:
    1. Movie title and release year
    2. Director and main cast (focusing on their favorite actors/directors when possible)
    3. Brief plot summary (no spoilers)
    4. Why they might like it based on their stated preferences
    5. Where they can likely watch it (streaming platforms, theaters, etc.)

    User Preferences:
    {preferences}

    Recent Search Results for Context:
    {context}

    Provide recommendations that match their favorite genres, actors, directors, and desired mood/themes.
    Include a mix of recent releases and classics based on their preference.
    """
    user_message = HumanMessage(content=user_prompt)

    # Use the LangChain ChatOpenAI instance
    response = chat_model.invoke([system_message, user_message])
    recommendations = response.content
    state["recommendations"] = recommendations

    # Evaluate the recommendations with JudgeVal
    judgment.async_evaluate(
        input=user_prompt,
        actual_output=recommendations,
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        model="gpt-4o"
    )

    return state

def should_continue_questions(state: State) -> bool:
    """Determine if we should continue asking questions."""
    return state["current_question_idx"] < len(state["questions"])

def router(state: State) -> str:
    """Route to the next node based on state."""
    if should_continue_questions(state):
        return "ask_question"
    return "search_movies"

# Build the workflow
workflow = StateGraph(State)

workflow.add_node("ask_question", ask_question)
workflow.add_node("process_answer", process_answer)
workflow.add_node("search_movies", search_movie_info)
workflow.add_node("generate_recommendations", generate_recommendations)

workflow.add_edge("ask_question", "process_answer")
workflow.add_conditional_edges(
    "process_answer",
    router,
    {
        "ask_question": "ask_question",
        "search_movies": "search_movies"
    }
)
workflow.add_edge("search_movies", "generate_recommendations")
workflow.add_edge("generate_recommendations", END)

workflow.set_entry_point("ask_question")

graph = workflow.compile()

def movie_recommendation_bot(handler: JudgevalCallbackHandler, query: str):
    """Main function to run the movie recommendation bot."""
    print("🎬 Welcome to the Movie Recommendation Bot! 🎬")
    print("I'll ask you a few questions to understand your movie taste, then suggest some films you might enjoy.")
    print("\nRunning with predefined answers for testing...\n")
    
    # Initialize state with predefined answers
    initial_state = initialize_state()
    
    try:
        # Run the entire workflow
        config_with_callbacks = {"callbacks": [handler]}
        final_state = graph.invoke(initial_state, config=config_with_callbacks)
        
        print("\n🍿 Your Personalized Movie Recommendations 🍿")
        print(final_state.get("recommendations", "No recommendations generated."))
        return final_state.get("recommendations", "No recommendations generated.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    handler = JudgevalCallbackHandler(judgment) 
    movie_recommendation_bot(handler, "Christopher Nolan")  # uncomment to run without the test (if you just want tracing)
    
    # This sets us up for running the unit test
    example = Example(
        input={"handler": handler, "query": "Christopher Nolan"},
        expected_tools=[
            {
                "tool_name": "search_movies",
                "parameters": {
                    "query": "Christopher Nolan"
                }
            }
        ]
    )
    
    client.assert_test(
        scorers=[ToolOrderScorer()],
        examples=[example],
        function=movie_recommendation_bot,
        eval_run_name="movie_langgraph_demo",
        project_name="movie_langgraph_demo"
    )