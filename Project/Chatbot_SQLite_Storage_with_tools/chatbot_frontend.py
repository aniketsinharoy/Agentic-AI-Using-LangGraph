import streamlit as st
from chatbot_backend import chatbot, get_past_conversations_thread_ids, get_thread_display_names
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessageChunk
import uuid

#--------------------------------------------------Utility Functions-----------------------------------------------------#
def generate_thread_id():
    return f"thread_{str(uuid.uuid4())}"

def reset_chat():
    st.session_state['messages_history'] = []
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['thread_history'].append(st.session_state['thread_id'])
    st.session_state['first_message_added'] = False
    # Don't add display name yet - wait for first user message

def load_conversation(thread_id):
    st.session_state['thread_id'] = thread_id
    st.session_state['messages_history'] = []  # Clear current messages
    state = chatbot.get_state({'configurable': {'thread_id': thread_id}})
    
    if state and hasattr(state, 'values') and 'messages' in state.values:
        for message in state.values['messages']:
            if message.type == 'human':
                st.session_state['messages_history'].append({'role': 'user', 'content': message.content})
            elif message.type == 'ai':
                st.session_state['messages_history'].append({'role': 'assistant', 'content': message.content})

#-------------------------------------------------Session State Initialization----------------------------------------------#

if 'messages_history' not in st.session_state:
    st.session_state['messages_history'] = []   

if 'thread_history' not in st.session_state:
    st.session_state['thread_history'] = get_past_conversations_thread_ids()  # Load past thread IDs from the backend

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['thread_history'].append(st.session_state['thread_id'])

if 'thread_display_names' not in st.session_state:
    st.session_state['thread_display_names'] = get_thread_display_names(st.session_state['thread_history'])  # Maps thread_id to display name (first message preview)

if 'first_message_added' not in st.session_state:
    st.session_state['first_message_added'] = False
    
#-------------------------------------------------------Variable Initialization----------------------------------------------#

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

#------------------------------------------------------------Streamlit UI----------------------------------------------------#

#---------Sidebar--------

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Chat History")

for thread_id in st.session_state['thread_history'][::-1]:  # Display threads in reverse order (most recent first)
    display_name = st.session_state['thread_display_names'].get(thread_id, "")
    
    if display_name:  # Only show button if thread has a display name
        if st.sidebar.button(display_name, key=thread_id):
            load_conversation(thread_id)

for message in st.session_state['messages_history']:
    with st.chat_message(message['role']):
        placeholder = st.empty()
        placeholder.markdown(message['content'])


#------------------Main Chat Area-----------------------

user_input = st.chat_input("Type your message here...")

if user_input:
    # If this is the first message for this thread, set the display name
    if st.session_state['thread_id'] not in st.session_state['thread_display_names']:
        display_name = " ".join(user_input.split(" ")[:4])
        st.session_state['thread_display_names'][st.session_state['thread_id']] = display_name
        st.session_state['first_message_added'] = True
    
    st.session_state['messages_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
       
    with st.chat_message('assistant'):
        full_response = ""
        placeholder = st.empty()

        config = {
            'configurable': {
                'thread_id': st.session_state['thread_id']
            }
        }

        for chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode='messages'
        ):

            if isinstance(chunk, AIMessageChunk):

                if chunk.content and isinstance(chunk.content, str):
                    full_response += chunk.content
                    placeholder.markdown(full_response)

        ai_message = full_response

    # with st.chat_message('assistant'):

    #     placeholder = st.empty()

    #     response = chatbot.invoke(
    #         {'messages': [HumanMessage(content=user_input)]},
    #         config=CONFIG
    #     )

    #     ai_message = response['messages'][-1].content

    #     placeholder.markdown(ai_message)

    st.session_state['messages_history'].append({'role': 'assistant', 'content': ai_message})    

    if st.session_state['first_message_added']:
        st.session_state['first_message_added'] = False
        st.rerun()
            