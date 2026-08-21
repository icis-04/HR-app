import asyncio
import time
from pathlib import Path
import base64
import inngest
import requests
import streamlit as st
from dotenv import load_dotenv
import os
 
load_dotenv()
 
st.set_page_config(
    page_title="HR Assistant",
    page_icon="🧑‍💼",
    layout="centered",
    initial_sidebar_state="expanded",
)
 
# ---------------------------------------------------------------------------
# Light styling — keeps it clean and HR-appropriate without heavy custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .source-pill {
            display: inline-block;
            background-color: #EEF2F6;
            color: #1F4E78;
            border-radius: 999px;
            padding: 2px 12px;
            margin: 2px 4px 2px 0;
            font-size: 0.8rem;
        }
        .doc-pill {
            display: inline-block;
            background-color: #F0FBF4;
            color: #1B6E3C;
            border-radius: 8px;
            padding: 4px 10px;
            margin: 3px 0;
            font-size: 0.85rem;
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ---------------------------------------------------------------------------
# Client + config
# ---------------------------------------------------------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=True)
 
 
def inngest_api_base() -> str:
    # Production default is Inngest Cloud's real API. Only overridden if you
    # explicitly set INNGEST_API_BASE (e.g. for local testing against the dev server).
    return os.getenv("INNGEST_API_BASE", "https://api.inngest.com/v1")
 
 
# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str, "sources": list}
 
if "documents" not in st.session_state:
    st.session_state.documents = []  # list of filenames ingested this session
 
 
# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------
def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path
 
 
async def send_rag_ingest_event(pdf_path: Path, source_id: str) -> None:
    client = get_inngest_client()
    pdf_bytes = pdf_path.read_bytes()
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_base64": encoded,
                "source_id": source_id,
            },
        )
    )
 
 
async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k},
        )
    )
    return result[0]
 
 
def fetch_runs(event_id: str) -> list[dict]:
    url = f"{inngest_api_base()}/events/{event_id}/runs"
 
    # Inngest Cloud's API requires the signing key as a Bearer token.
    # Without this header, every request comes back 401 Unauthorized.
    signing_key = os.environ["INNGEST_SIGNING_KEY"]
    headers = {"Authorization": f"Bearer {signing_key}"}
 
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])
 
 
def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"The workflow run {status.lower()}. Check the Inngest dashboard for details.")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for a response (last status: {last_status}).")
        time.sleep(poll_interval_s)
 
 
# ---------------------------------------------------------------------------
# Sidebar — document upload + tracking
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 HR Documents")
    st.caption("Upload CVs, policies, or onboarding guides for the assistant to reference.")
 
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
 
    if uploaded is not None and uploaded.name not in st.session_state.documents:
        with st.spinner(f"Ingesting {uploaded.name}..."):
            try:
                path = save_uploaded_pdf(uploaded)
                asyncio.run(send_rag_ingest_event(path))
                st.session_state.documents.append(uploaded.name)
                st.success(f"Added: {uploaded.name}")
            except Exception as e:
                st.error(f"Couldn't ingest this file: {e}")
 
    st.divider()
 
    if st.session_state.documents:
        st.caption(f"{len(st.session_state.documents)} document(s) ingested this session")
        for doc in st.session_state.documents:
            st.markdown(f'<div class="doc-pill">✅ {doc}</div>', unsafe_allow_html=True)
    else:
        st.caption("No documents ingested yet.")
 
    st.divider()
 
    top_k = st.slider("Chunks to retrieve per question", min_value=1, max_value=20, value=5)
 
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
 
    st.caption(
        "Note: this list only tracks documents uploaded in your current browser session — "
        "the assistant can still see anything ingested previously."
    )
 
 
# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------
st.title("🧑‍💼 HR Assistant")
st.caption("Ask about policies, onboarding, or candidate CVs. Answers are grounded only in uploaded documents.")
 
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown(
                " ".join(f'<span class="source-pill">📎 {s}</span>' for s in msg["sources"]),
                unsafe_allow_html=True,
            )
 
question = st.chat_input("Ask a question about your HR documents...")
 
if question:
    st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.write(question)
 
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking..._")
        try:
            event_id = asyncio.run(send_rag_query_event(question, top_k))
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "").strip()
            sources = output.get("sources", [])
            num_contexts = output.get("num_contexts", 0)
 
            placeholder.write(answer or "I wasn't able to generate an answer for that.")
 
            if sources:
                st.markdown(
                    " ".join(f'<span class="source-pill">📎 {s}</span>' for s in sources),
                    unsafe_allow_html=True,
                )
            elif num_contexts == 0:
                st.caption(
                    "No matching content was found in the uploaded documents for this question."
                )
 
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
 
        except TimeoutError as e:
            placeholder.error(str(e))
        except RuntimeError as e:
            placeholder.error(str(e))
        except requests.exceptions.ConnectionError:
            placeholder.error(
                "Couldn't reach the Inngest API. Check your network connection and try again."
            )
        except Exception as e:
            placeholder.error(f"Something went wrong: {e}")
 










