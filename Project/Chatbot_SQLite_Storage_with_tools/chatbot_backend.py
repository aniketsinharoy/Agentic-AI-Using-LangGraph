from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Literal
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from dotenv import load_dotenv

import sqlite3
import requests
import os

load_dotenv()

#----------------------------------------tools----------------------------------
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_number: float, second_number: float, operation: Literal["add", "subtract", "multiply", "divide"]) -> float:
    """Performs basic arithmetic operations on two numbers.
    Args:
        first_number (float): The first number.
        second_number (float): The second number.
        operation (str): The operation to perform. Can be "add", "subtract", "multiply", or "divide".
    """
    if operation == "add":
        return first_number + second_number
    elif operation == "subtract":
        return first_number - second_number
    elif operation == "multiply":
        return first_number * second_number
    elif operation == "divide":
        return first_number / second_number
    else:
        raise ValueError("Invalid operation")

STOCK_PRICE_API_KEY = os.getenv("STOCK_PRICE_API_KEY")

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetches the current stock price for a given symbol using a public API."""
    api_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_PRICE_API_KEY}'
    response = requests.get(api_url)
    return response.json()

EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

@tool
def get_exchange_rate_scaling_factor(from_currency: str, to_currency: str) -> float:
    """Takes two currency codes and returns the exchange rate scaling factor from the first currency to the second."""

    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    conversion_rate = response.json()['conversion_rate']
    
    return conversion_rate

#------------------------------------------------------------------------------------------------------------
    
#Make tool list
tool_list = [search_tool, calculator, get_stock_price, get_exchange_rate_scaling_factor] 

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(model="qwen/qwen3-32b", temperature=0)
#model = ChatCohere(model='command-a-03-2025')

#Bind tools with LLM
model_with_tools = model.bind_tools(tool_list, parallel_tool_calls=False)

def chatbot_function(state: ChatState) -> dict:

        messages = state['messages']
        response = model_with_tools.invoke(messages)
        return {'messages': [response]}

tool_node = ToolNode(tool_list)

graph = StateGraph(ChatState)

conn = sqlite3.connect(database='chatbot_conversations.db', check_same_thread=False)  # Ensure the database file is created
checkpointer = SqliteSaver(conn=conn)    

graph.add_node('chatbot', chatbot_function)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chatbot')
graph.add_conditional_edges('chatbot', tools_condition)
graph.add_edge('tools', 'chatbot')

chatbot = graph.compile(checkpointer=checkpointer)

# response = chatbot.invoke(
#     {'messages': [HumanMessage(content="What is the current stock price of Infosys in INR? How much money do I need to buy 10 shares?")]},
#     config={'configurable': {'thread_id': 'thread_1'}}
# )
# print(response['messages'][-1].content)

def get_past_conversations_thread_ids():
    thread_ids_set = set()
    for checkpoint in checkpointer.list(None):
        thread_ids_set.add(checkpoint.config['configurable']['thread_id'])
    return list(thread_ids_set)

def get_thread_display_names(thread_ids):
    display_name_thread_id_map = {}
    for thread_id in thread_ids:
        for checkpoint in checkpointer.list(config={'configurable': {'thread_id': thread_id}}): 
            if 'messages' in checkpoint.checkpoint['channel_values']:
                display_name = " ".join(checkpoint.checkpoint['channel_values']['messages'][0].content.split(" ")[:4])
                display_name_thread_id_map[thread_id] = display_name
                break
    return display_name_thread_id_map