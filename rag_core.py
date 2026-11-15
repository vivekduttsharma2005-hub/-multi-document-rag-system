# rag_core.py
# main logic of the RAG system (reading pdfs, searching, asking the LLM)

import re
import time
import numpy as np
from pypdf import PdfReader
from groq import NotFoundError, RateLimitError

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5            # chunks picked by meaning (semantic search)
KEYWORD_K = 6        # extra chunks picked by exact words (keyword search)
MIN_SCORE = 0.25     # if best match is lower than this, say not found
MEMORY_TURNS = 4     # how many previous Q&A we remember
# models are tried in this order (if one is removed by groq, the next one is used)
LLM_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
NOT_FOUND = "I could not find this information in the uploaded documents."

model = None


def get_model():
    # load the embedding model only once
    global model
    if model is None:
        # imported here so the app page opens fast (this library is heavy)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def split_text(text):
    # break the page text into small overlapping pieces
    text = " ".join(text.split())
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + CHUNK_SIZE])
        start = start + CHUNK_SIZE - CHUNK_OVERLAP
    return pieces


def build_index(pdf_list):
    # pdf_list = list of (file name, file path or uploaded file)
    chunks = []
    for name, pdf in pdf_list:
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for piece in split_text(text):
                if len(piece.strip()) > 50:   # skip very small pieces
                    chunks.append({"text": piece, "source": name, "page": page_num})

    if len(chunks) == 0:
        return [], None

    texts = [c["text"] for c in chunks]
    embeddings = get_model().encode(texts, normalize_embeddings=True)
    return chunks, np.array(embeddings)


STOP_WORDS = set(
    "the a an of in on for to and or is was were what which who how did does do it its this that "
    "these those are be by with from as at about report mentioned previous year compared change".split()
)


def get_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def keyword_scores(query, chunks):
    # simple word matching: +1 for each query word found, +2 for each pair of words found together
    words = get_words(query)
    unique = list(dict.fromkeys(words))
    pairs = [a + " " + b for a, b in zip(words, words[1:])]
    scores = []
    for c in chunks:
        text = c["text"].lower()
        score = 0
        for w in unique:
            if re.search(r"\b" + re.escape(w) + r"\b", text):
                score += 1
        for p in pairs:
            if p in text:
                score += 2
        scores.append(score)
    return np.array(scores)


def search(query, chunks, embeddings):
    # semantic search alone misses numbers and tables, so we also use keyword search
    q_emb = get_model().encode([query], normalize_embeddings=True)[0]
    sem = embeddings @ q_emb                       # cosine similarity
    picked = list(np.argsort(sem)[::-1][:TOP_K])   # best by meaning (highest score first)

    kw = keyword_scores(query, chunks)
    for i in np.argsort(kw)[::-1][:KEYWORD_K]:
        if kw[i] >= 2 and i not in picked:
            picked.append(i)

    return [(chunks[i], float(sem[i])) for i in picked]


def ask_llm(client, messages, temperature=0.1):
    last_error = None
    for model_name in LLM_MODELS:
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model=model_name, messages=messages, temperature=temperature
                )
                return (response.choices[0].message.content or "").strip()
            except RateLimitError as e:
                # free tier has a tokens-per-minute limit, wait and try again
                last_error = e
                time.sleep(5 * (attempt + 1))
            except NotFoundError as e:
                last_error = e      # model not available, try the next one
                break
    raise last_error


def get_history(messages):
    # only last few messages so the prompt doesnt get too long
    last = messages[-2 * MEMORY_TURNS:]
    lines = []
    for m in last:
        lines.append(m["role"].capitalize() + ": " + m["content"])
    return "\n".join(lines)


def make_standalone(client, question, messages):
    # follow up questions like "what about last year?" dont make sense alone,
    # so we ask the LLM to rewrite them using the chat history
    if len(messages) == 0:
        return question
    prompt = (
        "Rewrite the follow-up question so that it can be understood without the chat history. "
        "Only return the rewritten question.\n\n"
        "Chat history:\n" + get_history(messages) + "\n\nFollow-up question: " + question
    )
    return ask_llm(client, [{"role": "user", "content": prompt}], temperature=0)


def answer_question(client, question, chunks, embeddings, messages):
    new_question = make_standalone(client, question, messages)
    results = search(new_question, chunks, embeddings)

    # nothing matched properly
    if len(results) == 0 or results[0][1] < MIN_SCORE:
        return NOT_FOUND, []

    context = ""
    for chunk, score in results:
        context += "[" + chunk["source"] + ", page " + str(chunk["page"]) + "]\n"
        context += chunk["text"] + "\n\n"

    system_prompt = (
        "You are an assistant that answers questions using only the given context. "
        "Write the source like (file.pdf, p.3) with normal round brackets only. "
        "If only a part of the answer is in the context, give that part and clearly say what is missing. "
        "Only if nothing in the context is related to the question, reply exactly: " + NOT_FOUND
    )
    user_prompt = (
        "Chat history:\n" + get_history(messages) +
        "\n\nContext:\n" + context +
        "\nQuestion: " + new_question
    )
    answer = ask_llm(client, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    if NOT_FOUND in answer:
        return answer, []

    sources = []
    for chunk, score in results:
        sources.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "score": round(score, 2),
            "text": chunk["text"],
        })
    return answer, sources

