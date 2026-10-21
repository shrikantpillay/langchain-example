import json
import truststore

truststore.inject_into_ssl()

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# STEP 1: LOAD API KEY
# ============================================================

load_dotenv()


# ============================================================
# STEP 2: LOAD KNOWLEDGE OBJECTS
# ============================================================

KNOWLEDGE_FOLDER = Path("knowledge")

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


# Convert Python objects into text for the prompt
knowledge = json.dumps(
    knowledge_objects,
    indent=2
)


print(
    f"Loaded {len(knowledge_objects)} "
    f"knowledge objects."
)


# ============================================================
# STEP 3: INITIALIZE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# STEP 4: DEFINE PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an IT Service Desk Engineer.

        You MUST answer the user's question using ONLY the
        information contained in the KNOWLEDGE BASE below.

        Do NOT add troubleshooting steps from your own knowledge.

        The knowledge has been derived from historical customer
        support tickets.

        Prefer resolution actions with stronger historical
        evidence.

        If the knowledge base does not contain enough information
        to answer the question, respond:

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
# STEP 5: CREATE LANGCHAIN
# ============================================================

chain = prompt | llm


# ============================================================
# STEP 6: GET USER QUESTION
# ============================================================

issue = input(
    "\nDescribe your IT issue: "
)


# ============================================================
# STEP 7: ASK GPT
# ============================================================

response = chain.invoke({
    "knowledge": knowledge,
    "user_issue": issue
})


# ============================================================
# STEP 8: DISPLAY RESPONSE
# ============================================================

print("\nAI Service Desk Response:\n")

print(
    response.content
)