import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Load API key
load_dotenv()

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
        You are an experienced IT Service Desk Engineer.

        Your responsibilities:
        1. Understand the user's IT problem.
        2. Identify possible root causes.
        3. Recommend troubleshooting steps.
        4. Keep responses simple and structured.
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
    "user_issue": issue
})

# Step 7: Display response
print("\nAI Service Desk Response:\n")
print(response.content)