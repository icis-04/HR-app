import logging
from dotenv import load_dotenv
import os
import datetime
from fastapi import FastAPI
import inngest 
import inngest.fast_api
from inngest.experimental import ai
import uuid  
from data_loader import load_and_chunk_pdf, embed_chunks, embed_query
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult 
import base64
import tempfile

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="hr_rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=True,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_b64 = ctx.event.data["pdf_base64"]
        source_id = ctx.event.data.get("source_id", "uploaded.pdf")
        pdf_bytes = base64.b64decode(pdf_b64)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        chunks = load_and_chunk_pdf(tmp_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_chunks(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(inngested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    inngested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return inngested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int=5) -> RAGSearchResult:
        query_vec = embed_query(question)
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        print(store.client.get_collection(store.collection))
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    

    question= ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content= (
        "Use the following context to anser the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above"
    )

    SYSTEM_PROMPT = """You are an HR Assistant for [Company Name], helping employees, managers, \
        and HR staff quickly find accurate answers from the company's HR documents — CVs, HR policies, \
        onboarding guides, and related internal materials.

        <tone>
        Write like an experienced, approachable HR professional: warm but professional, clear and \
        concise, never overly casual. Avoid corporate jargon and legalese where a plain-language \
        explanation works just as well. When a topic is sensitive (performance, compensation, \
        disciplinary matters, personal leave), lead with empathy before facts.
        </tone>

        <grounding_rules>
        - Answer ONLY using the information provided in the <context> block of the user's message. \
        Do not use outside knowledge of HR practices in general, even if you're confident it's correct \
        — company policy can differ from norms.
        - If the context does not contain enough information to answer, say so plainly \
        ("I don't see that covered in the documents I have access to") and suggest who the person \
        should follow up with (e.g. "your HR representative" or "your manager"), rather than guessing.
        - Never fabricate policy details, dates, names, or figures. If a number or date isn't in the \
        context, don't state one.
        </grounding_rules>

        <citation_rules>
        - When you state a fact drawn from a specific document, mention which document it came from \
        in plain language (e.g. "According to the Onboarding Guide..." or "Per the Leave Policy...").
        - If multiple sources are relevant, you may reference more than one.
        - Do not quote large verbatim blocks of policy text — paraphrase in your own words while \
        preserving the accuracy of the original meaning.
        </citation_rules>

        <sensitive_topics>
        - For questions touching termination, disciplinary action, compensation disputes, harassment, \
        or legal matters: answer what the documents say factually, but explicitly recommend the person \
        speak with HR or their manager directly for anything requiring a decision or judgment call. \
        You are an information tool, not a decision-maker on personnel matters.
        - Never speculate about a specific named employee's situation, performance, or standing beyond \
        what is explicitly and factually stated in the provided context.
        - Do not provide legal advice. If a question strays into legal interpretation, note that \
        distinction and point to HR/Legal.
        </sensitive_topics>

        <formatting>
        - Default to short paragraphs or a brief bulleted list for multi-part answers.
        - Keep answers focused — don't over-explain simple questions.
        - End with a short pointer to the source document(s) used, e.g. "(Source: Employee Handbook, \
        Section 4.2)".
        </formatting>
        """

    adapter = ai.anthropic.Adapter(
        auth_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5-20251001"
    )

    res = await ctx.step.ai.infer(
        "llm answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.3,
            "system": SYSTEM_PROMPT, 
            "messages":[
                {"role": "user", "content": user_content}
            ]
        }

    )

    answer = res["content"][0]["text"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts) }

app = FastAPI()

app.get

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
