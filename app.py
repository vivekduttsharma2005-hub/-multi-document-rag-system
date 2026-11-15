import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from rag_core import build_index, answer_question
from chat_store import new_session_id, load_all, save_session, get_session, delete_session

load_dotenv()   
api_key = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="Multi-Document RAG", layout="wide")
st.title("Multi-Document RAG System")
st.write("Upload your PDF reports and ask questions about them.")

if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.embeddings = None


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown("**" + s["source"] + "** - page " + str(s["page"]) +
                        " (score: " + str(s["score"]) + ")")
            st.caption(s["text"][:300] + "...")


def start_new_chat():
    st.session_state.session_id = new_session_id()
    st.session_state.messages = []


# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Process documents") and files:
        with st.spinner("Reading PDFs..."):
            pdf_list = [(f.name, f) for f in files]
            chunks, embeddings = build_index(pdf_list)
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
        if len(chunks) > 0:
            st.success(str(len(files)) + " file(s) processed, " + str(len(chunks)) + " chunks")
        else:
            st.error("Could not read any text from these PDFs")

    if len(st.session_state.chunks) > 0:
        st.write("Loaded documents:")
        for name in sorted(set(c["source"] for c in st.session_state.chunks)):
            st.write("- " + name)

    st.divider()
    st.header("Chat session")
    st.write("Current session ID: `" + st.session_state.session_id + "`")

    if st.button("New chat"):
        start_new_chat()
        st.rerun()

    # old chats
    saved = load_all()
    if len(saved) > 0:
        st.write("Previous chats")
        ids = sorted(saved.keys(), key=lambda s: saved[s]["created"], reverse=True)
        chosen = st.selectbox(
            "Select a chat", ids,
            format_func=lambda s: s + " | " + saved[s]["created"] + " | " + saved[s]["title"],
            label_visibility="collapsed",
        )
        col1, col2 = st.columns(2)
        if col1.button("Open"):
            st.session_state.session_id = chosen
            st.session_state.messages = get_session(chosen)
            st.rerun()
        if col2.button("Delete"):
            delete_session(chosen)
            if chosen == st.session_state.session_id:
                start_new_chat()
            st.rerun()
        st.caption("Note: to ask new questions in an old chat, process the PDFs again.")

# ---------- old messages ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# ---------- new question ----------
question = st.chat_input("Ask something about your documents")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    if api_key == "":
        st.error("API key not found. Add GROQ_API_KEY in the .env file and restart the app.")
    elif len(st.session_state.chunks) == 0:
        st.warning("Upload PDFs and click Process documents first")
    else:
        client = Groq(api_key=api_key)
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                answer, sources = answer_question(
                    client, question,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    st.session_state.messages,
                )
            st.markdown(answer)
            if sources:
                show_sources(sources)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        save_session(st.session_state.session_id, st.session_state.messages)
        st.rerun()
