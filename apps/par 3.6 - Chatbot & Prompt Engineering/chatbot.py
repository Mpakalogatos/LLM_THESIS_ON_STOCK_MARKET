import streamlit as st
import requests
from ddgs import DDGS
import time

# --- Ollama wrapper ---
def llama3_ollama(prompt, model="llama3", timeout=1000):
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        answer = f"⚠️ Connection error: {e}"
    except ValueError:
        answer = "⚠️ Invalid JSON response from Ollama."
    elapsed = time.time() - start_time
    return answer, elapsed

# --- Web search ---
def search_web(query, num_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        if not results:
            return "No search results found."
        snippets = "\n\n".join([
            f"Title: {r.get('title','No title')}\nSnippet: {r.get('body','')[:300]}...\nURL: {r.get('href','')}"
            for r in results
        ])
        return snippets
    except Exception as e:
        return f"⚠️ Search error: {e}"

# --- Ask LLaMA without history ---
def ask_stateless(question, model="llama3"):
    search_results = search_web(question)
    prompt = (
        "You are a helpful AI assistant.\n"
        f"Use the following web search results to answer the user's question:\n{search_results}\n\n"
        f"User's question: {question}\n"
        "Answer concisely and cite URLs when possible."
    )
    return llama3_ollama(prompt, model=model)

# --- Streamlit UI ---
st.set_page_config(page_title="🌐 LLaMA 3 Web Chat", layout="wide")
st.title("🌐 LLaMA 3 Web Chat")

question = st.text_area("Ask a question:", height=300)
model = st.selectbox("Choose a local Ollama model:", ["llama3", "llama3.1", "mistral", "gemma2"])

if st.button("Ask") and question.strip():
    with st.spinner("🔎 Searching and thinking..."):
        answer, elapsed = ask_stateless(question, model=model)
        st.markdown(f"**Answer:** {answer}")
        st.markdown(f"_Took {elapsed:.2f} seconds_")
