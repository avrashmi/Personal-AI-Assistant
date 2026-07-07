import streamlit as st
from single_file.agent import graph

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖")

st.title("🤖 Personal AI Assistant")
st.write("Ask questions about your calendar or to-do list.")


# -------------------------
# Session Memory
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# Display chat
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------
# User input
# -------------------------

user_input = st.chat_input("Ask something like: What meetings do I have tomorrow?")


if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    # Call graph with memory
    result = graph.invoke({
        "question": user_input,
        "chat_history": st.session_state.messages
    })

    answer = result["answer"]

    # Update session memory
    st.session_state.messages = result["chat_history"]

    with st.chat_message("assistant"):
        st.write(answer)