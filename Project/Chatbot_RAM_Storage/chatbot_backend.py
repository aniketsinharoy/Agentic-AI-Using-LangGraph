from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(model='llama-3.3-70b-versatile')

def chatbot_function(state: ChatState) -> dict:

        messages = state['messages']
        response = model.invoke(messages)
        return {'messages': [response]}

graph = StateGraph(ChatState)
checkpointer = MemorySaver()    #Used to save the state in RAM

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