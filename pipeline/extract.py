# ============================================================
# IMPORTS
# ============================================================

import json
import re
import truststore
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# STEP 1: SETUP
# ============================================================

truststore.inject_into_ssl()
load_dotenv()


# ============================================================
# STEP 2: DEFINE PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "labeled_tickets.csv"
)

KNOWLEDGE_FOLDER = (
    PROJECT_ROOT
    / "knowledge"
)

# Create knowledge folder if it doesn't exist
KNOWLEDGE_FOLDER.mkdir(
    exist_ok=True
)


# ============================================================
# STEP 3: LOAD LABELED TICKETS
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n--- LABELED DATA LOADED ---")
print(f"Tickets: {len(df)}")
print(f"Issue families: {df['issue_family'].nunique()}")


# ============================================================
# STEP 4: INITIALIZE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ============================================================
# STEP 5: CREATE RESOLUTION EXTRACTION PROMPT
# ============================================================

extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an IT Service Management knowledge analyst.

        You will receive historical support tickets belonging
        to the same issue family.

        For each ticket extract:

        - ticket_id
        - action
        - outcome

        Rules:

        1. Use ONLY information contained in the ticket.

        2. Do NOT invent technical actions or outcomes.

        3. Normalize technically equivalent actions to exactly
           the same action name across all tickets.

        4. Use short action-oriented wording.

        Examples:

        "Restarted PanGPS service"
        becomes:
        "Restart PanGPS service"

        "Recreated GlobalProtect profile"
        and
        "Recreated VPN profile"
        should both become:
        "Recreate VPN profile"

        5. If an outcome is not available, return null.

        6. Return ONLY valid JSON.

        Return a JSON list:

        [
            {{
                "ticket_id": "INC001",
                "action": "Restart PanGPS service",
                "outcome": "Connection restored"
            }}
        ]
        """
    ),
    (
        "human",
        """
        ISSUE FAMILY:

        {issue_family}

        HISTORICAL TICKETS:

        {tickets}
        """
    )
])


# ============================================================
# STEP 6: CREATE KNOWLEDGE SYNTHESIS PROMPT
# ============================================================

knowledge_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an IT Service Management knowledge analyst.

        Create reusable troubleshooting knowledge using
        historical customer support evidence.

        You MUST use ONLY the evidence provided.

        Do NOT invent:

        - root causes
        - technical actions
        - troubleshooting steps
        - validation steps

        Resolution patterns with higher historical frequency
        should appear earlier.

        Confidence rules:

        HIGH:
        Multiple historical tickets provide clear and
        consistent resolution evidence.

        MEDIUM:
        Some reusable evidence exists but evidence is
        limited or inconsistent.

        LOW:
        Historical evidence is insufficient for reliable
        reusable knowledge.

        Return ONLY valid JSON.

        Use this structure:

        {{
            "issue": "...",
            "historical_ticket_count": 0,
            "recommended_troubleshooting": [
                {{
                    "step": 1,
                    "action": "...",
                    "evidence_count": 0,
                    "percentage": 0
                }}
            ],
            "confidence": "HIGH/MEDIUM/LOW"
        }}
        """
    ),
    (
        "human",
        """
        ISSUE FAMILY:

        {issue_family}

        TOTAL HISTORICAL TICKETS:

        {ticket_count}

        OBSERVED RESOLUTION PATTERNS:

        {patterns}
        """
    )
])


# ============================================================
# STEP 7: CREATE LANGCHAINS
# ============================================================

extraction_chain = (
    extraction_prompt
    | llm
)

knowledge_chain = (
    knowledge_prompt
    | llm
)


# ============================================================
# STEP 8: FUNCTION TO CREATE SAFE FILE NAME
# ============================================================

def create_filename(issue_family):

    filename = issue_family.lower()

    # Replace spaces and special characters with underscore
    filename = re.sub(
        r"[^a-z0-9]+",
        "_",
        filename
    )

    # Remove leading/trailing underscores
    filename = filename.strip("_")

    return filename + ".json"


# ============================================================
# STEP 9: PROCESS EVERY ISSUE FAMILY
# ============================================================

issue_families = df[
    "issue_family"
].unique()


for issue_family in issue_families:

    print(
        f"\n========================================"
    )

    print(
        f"PROCESSING: {issue_family}"
    )

    print(
        f"========================================"
    )


    # --------------------------------------------------------
    # 9A: SELECT TICKETS FOR THIS ISSUE FAMILY
    # --------------------------------------------------------

    issue_tickets = df[
        df["issue_family"] == issue_family
    ]

    total_tickets = len(
        issue_tickets
    )

    print(
        f"Historical tickets: {total_tickets}"
    )


    # --------------------------------------------------------
    # 9B: PREPARE TICKETS FOR GPT
    # --------------------------------------------------------

    ticket_data = issue_tickets[
        [
            "ticket_id",
            "description",
            "resolution_comment"
        ]
    ].to_dict(
        orient="records"
    )

    ticket_text = json.dumps(
        ticket_data,
        indent=2
    )


    # --------------------------------------------------------
    # 9C: EXTRACT RESOLUTION ACTIONS
    # --------------------------------------------------------

    extraction_response = extraction_chain.invoke({
        "issue_family": issue_family,
        "tickets": ticket_text
    })

    extracted_actions = json.loads(
        extraction_response.content
    )

    actions_df = pd.DataFrame(
        extracted_actions
    )


    print("\nExtracted Actions:")

    print(
        actions_df[
            [
                "ticket_id",
                "action"
            ]
        ]
    )


    # --------------------------------------------------------
    # 9D: COUNT RESOLUTION PATTERNS
    # --------------------------------------------------------

    action_counts = (
        actions_df["action"]
        .value_counts()
        .reset_index()
    )

    action_counts.columns = [
        "action",
        "ticket_count"
    ]


    # --------------------------------------------------------
    # 9E: CALCULATE PERCENTAGES
    # --------------------------------------------------------

    action_counts["percentage"] = (
        action_counts["ticket_count"]
        / total_tickets
        * 100
    ).round(1)


    print("\nResolution Patterns:")

    print(action_counts)


    # --------------------------------------------------------
    # 9F: PREPARE PATTERNS FOR KNOWLEDGE SYNTHESIS
    # --------------------------------------------------------

    pattern_data = action_counts.to_dict(
        orient="records"
    )

    pattern_text = json.dumps(
        pattern_data,
        indent=2
    )


    # --------------------------------------------------------
    # 9G: GENERATE CONSOLIDATED KNOWLEDGE
    # --------------------------------------------------------

    knowledge_response = knowledge_chain.invoke({
        "issue_family": issue_family,
        "ticket_count": total_tickets,
        "patterns": pattern_text
    })

    knowledge = json.loads(
        knowledge_response.content
    )


    # --------------------------------------------------------
    # 9H: DISPLAY KNOWLEDGE
    # --------------------------------------------------------

    print("\nConsolidated Knowledge:")

    print(
        json.dumps(
            knowledge,
            indent=2
        )
    )


    # --------------------------------------------------------
    # 9I: SAVE KNOWLEDGE OBJECT
    # --------------------------------------------------------

    filename = create_filename(
        issue_family
    )

    output_file = (
        KNOWLEDGE_FOLDER
        / filename
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            knowledge,
            file,
            indent=2
        )


    print(
        f"\nKnowledge saved: {output_file}"
    )


# ============================================================
# STEP 10: COMPLETE
# ============================================================

print("\n========================================")
print("KNOWLEDGE FACTORY COMPLETED")
print("========================================")

print(
    f"Knowledge objects created: "
    f"{len(issue_families)}"
)

print(
    f"Location: {KNOWLEDGE_FOLDER}"
)