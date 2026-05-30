from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_cohere import ChatCohere

from dotenv import load_dotenv

load_dotenv()

class MainGraphState(TypedDict):
    topic: str
    joke: str
    explanation: str

#-----------------------------Subgraph to explain a joke----------------------------------

subgraph_model = ChatCohere(model="command-a-03-2025")

def explain_joke(state: MainGraphState) -> dict:
    joke = state['joke']
    prompt = f"Explain the following joke in simple terms: {joke}"
    response = subgraph_model.invoke(prompt)
    print(f"[Subgraph] Explanation: {response.content}")
    return {"explanation": response.content}

subgraph = StateGraph(MainGraphState)

subgraph.add_node("explain_joke", explain_joke)

subgraph.add_edge(START, "explain_joke")
subgraph.add_edge("explain_joke", END)

subgraph_workflow = subgraph.compile()

#-----------------------------Main graph to invoke the subgraph----------------------------------

maingraph_model = ChatCohere(model="command-a-plus-05-2026")

def create_joke(state: MainGraphState) -> dict:
    topic = state['topic']
    prompt = f"Tell me a joke about {topic}"
    response = maingraph_model.invoke(prompt)

    #ChatCohere new model returns a list of items when we try to print response.content, so we need to extract the text from the response
    for item in response.content:
        if item["type"] == "text":
            joke = item["text"]
            break

    print(f"[Main Graph] Generated joke: {joke}")
    return {"joke": joke}

main_graph = StateGraph(MainGraphState)

main_graph.add_node("create_joke", create_joke)
main_graph.add_node("explain_joke_in_subgraph", subgraph_workflow)

main_graph.add_edge(START, "create_joke")
main_graph.add_edge("create_joke", "explain_joke_in_subgraph")
main_graph.add_edge("explain_joke_in_subgraph", END)

main_graph_workflow = main_graph.compile()


#------------------------Invoke the main graph------------------------------------------

topic = "technology"
final_response = main_graph_workflow.invoke({"topic": topic})
print(f"Final response: {final_response}")
