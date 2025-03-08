import os
from typing import TypedDict

from IPython.core.display import Image
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph

### LLM
local_llm = str(os.getenv("LLM_MODEL"))
llm = ChatOllama(model=local_llm, temperature=0.0)

### Tool utilities

email_agent_description = """
You are an AI Email Assistant, your primary task is to efficiently manage the user’s inbox by categorizing, drafting, prioritizing and responding to emails. You will mimic the user’s email writing style and detect important tasks. You are also responsible for summarizing email threads and sending follow-up reminders when necessary.

Tools:

1. **Email Summarization**: Summarize the incoming email into 1 to 5 bulletpoints
2. **Email Categorization**: Sort the incoming email into a category: urgent, follow-up, informational, spam, newsletter or other
3. **Email Analyzer**: Based on the category and the summary of the incoming email it will determine whether it requires a direct response, a later response, or no response
4. **Email Drafting**: Draft a response to the incoming email based on the summary and the category based on the user's writing style and send it if the category is urgent.
5. **Email Reminder**: Send a follow-up reminder to the user if the email is categorized as follow-up and the user has not responded within a certain time frame.
"""


email_summarization_instructions = """
You are an AI Email Summarizer, your primary task is to summarize incoming emails into 1 to 5 bullet points. You will extract the most important information from the email and present it in a concise manner.
"""

email_summarization_prompt = """
Summarize the incoming email into 1 to 5 bullet points.

Input:

{input}

Personal Data:

{personal_data}
"""

email_category_instructions = """
You are an AI Email Categorizer, your primary task is to categorize incoming emails into one of the following categories: urgent, follow-up, informational, spam, newsletter or other. You will analyze the content of the email and determine its priority and relevance.
"""

email_category_prompt = """
Categorize the incoming email into one of the following categories: urgent, follow-up, informational, spam, newsletter or other.

Input:

{input}
"""

email_analyzer_instructions = """
You are an AI Email Analyzer, your primary task is to analyze incoming emails based on the summary and category provided. You will determine whether the email requires a direct response, a later response, or no response.
"""

email_analyzer_prompt = """
Analyze the incoming email based on the summary and category provided.

Summary:

{summary}

Category:

{category}
"""

email_drafting_instructions = """
You are an AI Email Drafter, your primary task is to draft a response to the incoming email based on the summary and category provided. You will mimic the user's writing style and send the response if the category is urgent.
"""

email_drafting_prompt = """
Draft a response to the incoming email based on the summary and category provided.

Summary:

{summary}

Category:

{category}
"""

email_reminder_instructions = """
You are an AI Email Reminder, your primary task is to send a follow-up reminder to the user if the email is categorized as follow-up and the user has not responded within 24 hours.
"""

email_reminder_prompt = """
Send a follow-up reminder to the user if the email is categorized as follow-up and the user has not responded within a 24 hours.

Summary:

{summary}

Category:

{category}
"""

### State

class EmailState(TypedDict):
    input: str
    personal_data: str
    summary: str
    category: str
    response: str
    reminder: str


### Nodes

def email_summarization(state: EmailState) -> EmailState:
    pass


def email_category(state: EmailState) -> EmailState:
    pass


def email_analyzer(state: EmailState) -> EmailState:
    pass


def email_drafting(state: EmailState) -> EmailState:
    pass


def email_reminder(state: EmailState) -> EmailState:
    pass

email_agent = StateGraph(EmailState)

email_agent.add_node("email_summarization", email_summarization)
email_agent.add_node("email_category", email_category)
email_agent.add_node("email_analyzer", email_analyzer)
email_agent.add_node("email_drafting", email_drafting)
email_agent.add_node("email_reminder", email_reminder)

email_agent.set_entry_point("email_summarization")
email_agent.set_entry_point("email_category")
email_agent.add_edge("email_summarization", "email_analyzer")
email_agent.add_edge("email_category", "email_analyzer")
email_agent.add_conditional_edges("email_analyzer",
                                  lambda state: state["response"],
                                  {
                                      "direct_response": "email_drafting",
                                      "later_response": "email_reminder",
                                      "no_response": END
                                  })
email_agent.add_edge("email_drafting", END)
email_agent.add_edge("email_reminder", END)