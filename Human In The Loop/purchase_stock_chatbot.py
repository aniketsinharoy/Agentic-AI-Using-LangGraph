from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

from dotenv import load_dotenv

import requests
import os

load_dotenv()

#----------------------------------------tools----------------------------------
STOCK_PRICE_API_KEY = os.getenv("STOCK_PRICE_API_KEY")

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetches the current stock price for a given symbol using a public API."""
    api_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_PRICE_API_KEY}'
    response = requests.get(api_url)
    return response.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """Simulates purchasing a stock. In a real application, this would interact with a brokerage API."""

    decision = interrupt(f"Are you sure you want to purchase {quantity} shares of {symbol}? (yes/no)")

    if decision.lower() == 'yes':
        # Simulate a successful purchase
        return {
            "status": "success",
            "message": f"Purchased {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} cancelled.",
            "symbol": symbol,
            "quantity": quantity
        }

#------------------------------------------------------------------------------------------------------------
    
#Make tool list
tool_list = [get_stock_price, purchase_stock] 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(model="qwen/qwen3-32b", temperature=0)

#Bind tools with LLM
model_with_tools = model.bind_tools(tool_list, parallel_tool_calls=False)


def chatbot_function(state: ChatState) -> dict:

        messages = state['messages']
        response = model_with_tools.invoke(messages)
        return {'messages': [response]}

tool_node = ToolNode(tool_list)

graph = StateGraph(ChatState)

checkpointer = MemorySaver()

graph.add_node('chatbot', chatbot_function)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chatbot')
graph.add_conditional_edges('chatbot', tools_condition)
graph.add_edge('tools', 'chatbot')

chatbot = graph.compile(checkpointer=checkpointer)

while True:

    config = {'configurable': {'thread_id': 'thread_1'}}

    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit', 'e', 'q']:
        print("Exiting chatbot. Goodbye!")
        break
    
    #May hit an interrupt here
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=config
    )

    interrupt_response = response.get("__interrupt__",[])

    if interrupt_response:
        prompt_to_human = interrupt_response[0].value
        print(f"HITL: {prompt_to_human}")

        decision = input("Your decision (yes/no): ")

        response = chatbot.invoke(
            Command(resume=decision),
            config=config
        )

    messages = response['messages']
    print("Chatbot:", messages[-1].content)     