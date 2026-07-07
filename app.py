import streamlit as st
#from graph.builder import graph
from graph.builder import compiled_graph as graph
from dotenv import load_dotenv
from core.logger import logger


load_dotenv()

st.set_page_config(page_title="Personal AI Assistant", page_icon="🤖")

st.title("🤖 Personal AI Assistant")
st.write("Ask questions about your calendar or to-do list.")

# -------------------------
# Session Memory
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something like: What meetings do I have tomorrow?")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)


    try:
        result = graph.invoke({
            "question": user_input,
            "chat_history": st.session_state.messages
        })

        answer = result["answer"]

        st.session_state.messages = result["chat_history"]

        with st.chat_message("assistant"):
            st.write(answer)

        # ---- Monitoring ----
        if "latency" in result:
            st.caption(f"⏱ Response time: {result['latency']}s")

    except Exception as e:
        logger.error(f"App error: {e}")
        st.error("Something went wrong.")

   
   
   
   
    