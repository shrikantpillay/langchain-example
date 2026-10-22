# ============================================================
# IMPORTS
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel

from app import ask_helpdesk


# ============================================================
# CREATE API
# ============================================================

api = FastAPI(
    title="AI Service Desk API"
)


# ============================================================
# DEFINE INPUT FORMAT
# ============================================================

class HelpdeskRequest(BaseModel):

    issue: str


# ============================================================
# HEALTH CHECK
# ============================================================

@api.get("/")
def health_check():

    return {
        "status": "AI Service Desk is running"
    }


# ============================================================
# HELPDESK ENDPOINT
# ============================================================

@api.post("/ask")
def ask_question(
    request: HelpdeskRequest
):

    answer = ask_helpdesk(
        request.issue
    )

    return {
        "answer": answer
    }