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
# STEP 2: DEFINE OUR STANDARD SCHEMA
# --------------------------------------------------

CANONICAL_FIELDS = [
    "ticket_id",
    "issue_type",
    "description",
    "category",
    "created_at",
    "resolved_at",
    "resolution_time",
    "resolution_comment"
]


# --------------------------------------------------
# STEP 3: FIND AND LOAD CUSTOMER DATA
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "customer_tickets.csv"

df = pd.read_csv(DATA_FILE, sep="\t")

print("\n--- DATASET LOADED ---")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# STEP 4: TAKE 10 SAMPLE TICKETS
# --------------------------------------------------

sample_data = df.head(10).to_csv(index=False)

print("\n--- SAMPLE DATA ---")
print(sample_data)


# --------------------------------------------------
# STEP 5: INITIALIZE LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# STEP 6: CREATE MAPPING PROMPT
# --------------------------------------------------

mapping_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a data engineer specializing in IT Service Management.

        We are building a Knowledge Factory.

        Our standard canonical schema is:

        ticket_id
        issue_type
        description
        category
        created_at
        resolved_at
        resolution_time
        resolution_comment

        Analyze the customer ticket sample.

        Map the customer's columns to our canonical schema.

        Rules:

        1. Use both column names and sample values to understand
           what each customer field represents.

        2. Do not invent source columns.

        3. If a canonical field is not available in the customer
           data, return null.

        4. Return ONLY valid JSON.

        5. Do not provide explanations.

        Return the following structure:

        {{
            "ticket_id": "source column or null",
            "issue_type": "source column or null",
            "description": "source column or null",
            "category": "source column or null",
            "created_at": "source column or null",
            "resolved_at": "source column or null",
            "resolution_time": "source column or null",
            "resolution_comment": "source column or null"
        }}
        """
    ),
    (
        "human",
        """
        CUSTOMER DATA SAMPLE:

        {sample_data}
        """
    )
])


# --------------------------------------------------
# STEP 7: CREATE LANGCHAIN
# --------------------------------------------------

mapping_chain = mapping_prompt | llm


# --------------------------------------------------
# STEP 8: ASK GPT TO MAP CUSTOMER FIELDS
# --------------------------------------------------

mapping_response = mapping_chain.invoke({
    "sample_data": sample_data
})

print("\n--- AI PROPOSED MAPPING ---")
print(mapping_response.content)


# --------------------------------------------------
# STEP 9: CONVERT AI RESPONSE INTO PYTHON DICTIONARY
# --------------------------------------------------

mapping = json.loads(mapping_response.content)


# --------------------------------------------------
# STEP 10: VALIDATE THE MAPPING
# --------------------------------------------------

errors = []

for canonical_field, source_column in mapping.items():

    if source_column is not None:

        if source_column not in df.columns:

            errors.append(
                f"{source_column} does not exist in customer data"
            )


if errors:

    print("\n--- MAPPING ERRORS ---")

    for error in errors:
        print(error)

    raise ValueError("AI mapping validation failed.")


print("\nMapping validation passed.")


# --------------------------------------------------
# STEP 11: REVERSE THE MAPPING
# --------------------------------------------------

rename_mapping = {
    source: canonical
    for canonical, source in mapping.items()
    if source is not None
}


# --------------------------------------------------
# STEP 12: RENAME CUSTOMER COLUMNS
# --------------------------------------------------

normalized_df = df.rename(
    columns=rename_mapping
)


# --------------------------------------------------
# STEP 13: CONVERT DATE FIELDS
# --------------------------------------------------

if "created_at" in normalized_df.columns:

    normalized_df["created_at"] = pd.to_datetime(
        normalized_df["created_at"],
        errors="coerce"
    )


if "resolved_at" in normalized_df.columns:

    normalized_df["resolved_at"] = pd.to_datetime(
        normalized_df["resolved_at"],
        errors="coerce"
    )


# --------------------------------------------------
# STEP 14: CALCULATE RESOLUTION TIME
# --------------------------------------------------

if (
    "created_at" in normalized_df.columns
    and "resolved_at" in normalized_df.columns
):

    normalized_df["resolution_time"] = (
        normalized_df["resolved_at"]
        - normalized_df["created_at"]
    ).dt.total_seconds() / 60


# --------------------------------------------------
# STEP 15: KEEP ONLY OUR STANDARD FIELDS
# --------------------------------------------------

available_fields = [
    field
    for field in CANONICAL_FIELDS
    if field in normalized_df.columns
]

normalized_df = normalized_df[available_fields]


# --------------------------------------------------
# STEP 16: SAVE NORMALIZED DATA
# --------------------------------------------------

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "normalized_tickets.csv"
)

normalized_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n--- KNOWLEDGE FACTORY OUTPUT ---")
print(normalized_df.head())

print("\nNormalized dataset saved to:")
print(OUTPUT_FILE)