
import csv
import glob
import json
import os
import time
from dotenv import load_dotenv
from groq import Groq
from rag_core import build_index, answer_question, NOT_FOUND

load_dotenv()

DOCS_FOLDER = "docs"
QUESTIONS_FILE = "eval_questions.json"
RESULT_FILE = "evaluation_results.csv"


def percent(correct, total):
    if total == 0:
        return "n/a"
    return str(correct) + "/" + str(total) + " = " + str(round(100 * correct / total)) + "%"


api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("Please set GROQ_API_KEY first")
client = Groq(api_key=api_key)

pdf_paths = sorted(glob.glob(os.path.join(DOCS_FOLDER, "*.pdf")))
if len(pdf_paths) == 0:
    raise SystemExit("No PDFs found in the docs folder")

print("Reading", len(pdf_paths), "pdf(s)...")
chunks, embeddings = build_index([(os.path.basename(p), p) for p in pdf_paths])
print("Total chunks:", len(chunks), "\n")

with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = json.load(f)

# [correct, total] for each metric
answer_score = [0, 0]
source_score = [0, 0]
notfound_score = [0, 0]
memory_score = [0, 0]

results = []
history = []

for i, q in enumerate(questions, start=1):
    # start a fresh chat unless this question is a follow up
    if not q.get("follow_up"):
        history = []

    answer, sources = answer_question(client, q["question"], chunks, embeddings, history)
    history.append({"role": "user", "content": q["question"]})
    history.append({"role": "assistant", "content": answer})

    cited_files = sorted(set(s["source"] for s in sources))
    refused = NOT_FOUND in answer
    source_ok = None

    if q.get("not_in_docs"):
        # correct only if the system says it was not found
        answer_ok = refused
        notfound_score[1] += 1
        notfound_score[0] += int(answer_ok)
    else:
        # correct if all expected keywords are in the answer
        answer_ok = not refused
        for word in q.get("keywords", []):
            if word.lower() not in answer.lower():
                answer_ok = False
        # keywords_any = any one of these is enough (e.g. same number in different formats)
        if q.get("keywords_any"):
            if not any(w.lower() in answer.lower() for w in q["keywords_any"]):
                answer_ok = False
        answer_score[1] += 1
        answer_score[0] += int(answer_ok)

        if q.get("source"):
            # compare without ".pdf" so "report (1).pdf" still matches "report"
            expected = q["source"].replace(".pdf", "")
            source_ok = any(expected in f for f in cited_files)
            source_score[1] += 1
            source_score[0] += int(source_ok)

        if q.get("follow_up"):
            memory_score[1] += 1
            memory_score[0] += int(answer_ok)

    print("Q" + str(i) + ":", q["question"])
    print("   result:", "PASS" if answer_ok else "FAIL", "| sources:", cited_files)

    results.append({
        "id": i,
        "type": q.get("type", ""),
        "question": q["question"],
        "answer": answer,
        "cited_sources": "; ".join(cited_files),
        "answer_correct": answer_ok,
        "source_correct": source_ok,
    })
    time.sleep(8)   # free api has a tokens per minute limit, so we wait between questions

with open(RESULT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n----- Evaluation Summary -----")
print("Answer accuracy       :", percent(*answer_score))
print("Source accuracy       :", percent(*source_score))
print("Not-found handling    :", percent(*notfound_score))
print("Conversational memory :", percent(*memory_score))
print("Full results saved in", RESULT_FILE)
