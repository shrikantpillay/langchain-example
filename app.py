import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Load API key
load_dotenv()

# Step 2: Load Knowledge source
with open("knowledge.txt", "r") as file:
    knowledge = file.read()
# Step 2: Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Step 3: Define prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an IT Service Desk Engineer.

        You MUST answer the user's question using ONLY the
        information contained in the KNOWLEDGE BASE below.

        Do NOT add troubleshooting steps from your own knowledge.

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

# Step 4: Create LangChain
chain = prompt | llm

# Step 5: Get user input
issue = input("Describe your IT issue: ")

# Step 6: Invoke LLM
response = chain.invoke({
    "knowledge":knowledge,
    "user_issue": issue
})

# Step 7: Display response
print("\nAI Service Desk Response:\n")
print(response.content)