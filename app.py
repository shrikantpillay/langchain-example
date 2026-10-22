# ============================================================
# IMPORTS
# ============================================================

import json
import truststore

truststore.inject_into_ssl()

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# STEP 1: LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# STEP 2: FIND KNOWLEDGE FOLDER
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

KNOWLEDGE_FOLDER = (
    PROJECT_ROOT
    / "knowledge"
)


# ============================================================
# STEP 3: LOAD ALL KNOWLEDGE JSON FILES
# ============================================================

knowledge_objects = []

for file_path in KNOWLEDGE_FOLDER.glob("*.json"):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        knowledge_objects.append(
            json.load(file)
        )


print(
    f"Loaded {len(knowledge_objects)} "
    f"knowledge objects."
)


# ============================================================
# STEP 4: CONVERT KNOWLEDGE INTO TEXT
# ============================================================

knowledge = json.dumps(
    knowledge_objects,
    indent=2
)


# ============================================================
# STEP 5: INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# STEP 6: DEFINE THE PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an IT Service Desk Engineer.

        You MUST answer the user's question using ONLY the
        information contained in the KNOWLEDGE BASE below.

        Do NOT add troubleshooting steps from your own knowledge.

        The knowledge base has been generated from historical
        customer support tickets.

        Prefer resolution actions that have stronger historical
        evidence.

        If the knowledge base does not contain enough information
        to answer the user's question, respond:

        "I do not have enough information in the knowledge base
        to resolve this issue."

        KNOWLEDGE BASE:

        {knowledge}
        """
    ),
    (
        "human",
        "{user_issue}"
    )
])


# ============================================================
# STEP 7: CREATE LANGCHAIN
# ============================================================

chain = prompt | llm


# ============================================================
# STEP 8: CREATE REUSABLE HELPDESK FUNCTION
# ============================================================

def ask_helpdesk(issue):

    response = chain.invoke({
        "knowledge": knowledge,
        "user_issue": issue
    })

    return response.content


# ============================================================
# STEP 9: COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    issue = input(
        "\nDescribe your IT issue: "
    )

    answer = ask_helpdesk(
        issue
    )

    print(
        "\nAI Service Desk Response:\n"
    )

    print(answer)