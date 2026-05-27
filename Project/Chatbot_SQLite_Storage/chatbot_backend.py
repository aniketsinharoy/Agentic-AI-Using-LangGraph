from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(model='llama-3.3-70b-versatile')

def chatbot_function(state: ChatState) -> dict:

        messages = state['messages']
        response = model.invoke(messages)
        return {'messages': [response]}

graph = StateGraph(ChatState)

conn = sqlite3.connect(database='chatbot_conversations.db', check_same_thread=False)  # Ensure the database file is created
checkpointer = SqliteSaver(conn=conn)    

graph.add_node('chatbot', chatbot_function)

graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot', END)

chatbot = graph.compile(checkpointer=checkpointer)

#Streaming text got from the model 

# stream = chatbot.stream(
#     {'messages': [HumanMessage(content="Receipe of pasta with tomato sauce")]},
#     config={'configurable': {'thread_id': 'thread_1'}},
#     stream_mode='messages'
# )

# print(type(stream))  #GeneratorType

# for message_chunks, metadata in stream:
#     if message_chunks.content:
#         print(message_chunks.content, end=" ", flush=True)


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