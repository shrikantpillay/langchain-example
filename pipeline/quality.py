import pandas as pd
import json
import truststore

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# STEP 1: SETUP
# --------------------------------------------------

truststore.inject_into_ssl()
load_dotenv()


# --------------------------------------------------
# STEP 2: FIND PROJECT AND LOAD NORMALIZED DATA
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "normalized_tickets.csv"

df = pd.read_csv(DATA_FILE)

print("\n--- KNOWLEDGE FITNESS CHECK ---")
print(f"Tickets received: {len(df)}")


# --------------------------------------------------
# STEP 3: DEFINE OBVIOUS JUNK RESOLUTIONS
# --------------------------------------------------

JUNK_RESOLUTIONS = [
    "done",
    "fixed",
    "resolved",
    "closed",
    "completed",
    "user confirmed",
    "issue resolved"
]


# --------------------------------------------------
# STEP 4: BASIC PYTHON QUALITY CHECK
# --------------------------------------------------

def check_resolution(resolution):

    # No resolution available
    if pd.isna(resolution):
        return "LOW"

    # Convert resolution into clean lowercase text
    resolution = str(resolution).strip().lower()

    # Empty resolution
    if resolution == "":
        return "LOW"

    # Known junk resolution
    if resolution in JUNK_RESOLUTIONS:
        return "LOW"

    # Very short resolution
    if len(resolution) < 15:
        return "LOW"

    # Not obviously bad
    return "POTENTIAL"


# Apply the function to every ticket
df["knowledge_quality"] = df["resolution_comment"].apply(
    check_resolution
)


# --------------------------------------------------
# STEP 5: SHOW BASIC FILTER RESULTS
# --------------------------------------------------

print("\n--- BASIC QUALITY FILTER ---")

print(
    df[
        [
            "ticket_id",
            "description",
            "resolution_comment",
            "knowledge_quality"
        ]
    ]
)


# --------------------------------------------------
# STEP 6: KEEP ONLY POTENTIAL KNOWLEDGE
# --------------------------------------------------

potential_tickets = df[
    df["knowledge_quality"] == "POTENTIAL"
]

print("\nPotential knowledge tickets:")
print(len(potential_tickets))


# --------------------------------------------------
# STEP 7: INITIALIZE GPT-OSS
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# STEP 8: CREATE KNOWLEDGE MINING PROMPT
# --------------------------------------------------

knowledge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an IT Service Management knowledge analyst.

        Your job is to determine whether a historical support
        ticket contains reusable troubleshooting knowledge.

        You MUST use ONLY the information provided in the ticket.

        Do NOT invent causes, actions, or validation steps.

        Extract the following:

        issue
        cause
        resolution
        validation
        quality

        Quality must be one of:

        HIGH
        MEDIUM
        LOW

        HIGH:
        The ticket contains a clear issue and a useful,
        reusable resolution.

        MEDIUM:
        The ticket contains some useful troubleshooting
        information, but important details may be missing.

        LOW:
        The ticket does not contain useful reusable knowledge.

        If cause, resolution, or validation information
        is not available in the ticket, return null.

        Return ONLY valid JSON.

        Use exactly this structure:

        {{
            "issue": "...",
            "cause": "...",
            "resolution": "...",
            "validation": "...",
            "quality": "HIGH/MEDIUM/LOW"
        }}
        """
    ),
    (
        "human",
        """
        ISSUE DESCRIPTION:

        {description}


        RESOLUTION COMMENT:

        {resolution_comment}
        """
    )
])


# --------------------------------------------------
# STEP 9: CREATE LANGCHAIN
# --------------------------------------------------

knowledge_chain = knowledge_prompt | llm


# --------------------------------------------------
# STEP 10: FUNCTION TO MINE ONE TICKET
# --------------------------------------------------

def mine_knowledge(row):

    response = knowledge_chain.invoke({
        "description": row["description"],
        "resolution_comment": row["resolution_comment"]
    })

    knowledge = json.loads(response.content)

    return knowledge


# --------------------------------------------------
# STEP 11: TEST KNOWLEDGE MINING
# --------------------------------------------------

if len(potential_tickets) > 0:

    # Take first potential ticket
    test_ticket = potential_tickets.iloc[0]

    print("\n--- ORIGINAL TICKET ---")

    print("Ticket ID:")
    print(test_ticket["ticket_id"])

    print("\nDescription:")
    print(test_ticket["description"])

    print("\nResolution:")
    print(test_ticket["resolution_comment"])


    # Send ticket to GPT
    knowledge = mine_knowledge(test_ticket)


    # Display AI extracted knowledge
    print("\n--- AI EXTRACTED KNOWLEDGE ---")

    print(
        json.dumps(
            knowledge,
            indent=2
        )
    )

else:

    print("\nNo potential knowledge tickets found.")