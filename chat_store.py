import json
import os
import uuid
from datetime import datetime

HISTORY_FILE = "chat_history.json"


def new_session_id():
    
    return uuid.uuid4().hex[:8]


def load_all():
    
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}     # file is broken or empty, start fresh


def write_all(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_session(session_id, messages):
    if len(messages) == 0:
        return
    data = load_all()
    old = data.get(session_id, {})
    first_question = messages[0]["content"]
    data[session_id] = {
        "title": first_question[:40],
        "created": old.get("created", datetime.now().strftime("%d %b %Y, %H:%M")),
        "messages": messages,
    }
    write_all(data)


def get_session(session_id):
    return load_all().get(session_id, {}).get("messages", [])


def delete_session(session_id):
    data = load_all()
    if session_id in data:
        del data[session_id]
        write_all(data)
