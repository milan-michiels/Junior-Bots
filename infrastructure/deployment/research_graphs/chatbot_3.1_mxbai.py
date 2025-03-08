import json
import logging
import operator
import os
from typing import Annotated
from typing import List

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

### LLM
local_llm = "llama3.1:8b"
llm = ChatOllama(model=local_llm, temperature=0, base_url=os.getenv("OLLAMA_URI"))
llm_json_mode = ChatOllama(
    model=local_llm, temperature=0, format="json", base_url=os.getenv("OLLAMA_URI")
)
embedder = OllamaEmbeddings(model="mxbai-embed-large", base_url=os.getenv("OLLAMA_URI"))

### Retriever
client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT"))
)
vectorstore = Chroma(
    collection_name="rummikub_rules_mxbai",
    client=client,
    embedding_function=embedder,
    create_collection_if_not_exists=False,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

## Components

### Router

# Prompt
router_instructions = """You are a routing assistant that determines the best data source to answer user questions. There are three options:

Vectorstore: Use this for **specific and detailed questions** about the rules, setup, or gameplay of Rummikub.
Websearch: Use this for **recent changes to Rummikub rules**, **current events**, or information explicitly marked as being **outside the scope** of the vectorstore.
Irrelevant: Use this for **questions unrelated to Rummikub**, including general questions about unrelated topics, technologies, or concepts or uncertain questions.

**Output Format:**
Respond strictly in the following JSON format:

{"datasource": "<vectorstore | websearch | irrelevant>"}

**Guidelines:**
1. Use `vectorstore` **only** for questions directly referencing Rummikub rules, setup, or gameplay. Examples:
   - "What are the rules for forming sets in Rummikub?"
   - "How many tiles does each player start with?"
2. Use `websearch` **only** for questions requiring external research or recent updates about Rummikub. Examples:
   - "Who is the current world champion of Rummikub?"
   - "Have there been recent changes to Rummikub rules?"
3. Use `irrelevant` for all other questions, including general knowledge or unrelated topics. Examples:
   - "What are the main causes of global warming?"
   - "What are the types of agent memory?"

Always prioritize the most accurate category.
"""

### Retrieval Grader

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}.

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

### Generate

# Prompt
rag_prompt = """You are an assistant for question-answering tasks.

Here is the context to use to answer the question:

{context}

Carefully analyze the context above.

Now, review the user question:

{question}

Provide a clear, direct, and concise answer to the user's question, using only the information from the context. Avoid repeating long excerpts from the context verbatim unless necessary to clarify your response.

Make sure the answer is well-structured, easy to understand, and directly addresses the user's query.

Answer:"""


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


### Hallucination Grader

# Hallucination grader instructions
hallucination_grader_instructions = """

You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS.

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
"""

# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}.

Return JSON with one key, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS."""

### Answer Grader

# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score.

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
"""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}.

Return JSON with one key, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria."""

### Search

web_search_tool = TavilySearchResults(max_results=3)


### State


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents


### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logging.info("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents, "loop_step": 0}


def handle_irrelevant(state):
    """
    Handle irrelevant questions

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains response to irrelevant question
    """
    logging.info("---HANDLE IRRELEVANT---")
    return {
        "generation": "This question is not related to the Rummikub rules. Please ask something specific to the game."
    }


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logging.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    logging.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            logging.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            logging.info("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            continue

    if len(filtered_docs) == 0:
        # We set a flag to indicate that we want to run web search only if all documents are irrelevant
        web_search = "Yes"
    return {"documents": filtered_docs, "web_search": web_search}


def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """
    logging.info("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}


### Edges


def route_question(state):
    """
    Route question to web search or RAG or none

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    logging.info("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        logging.info("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        logging.info("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source == "irrelevant":
        logging.info("---ROUTE QUESTION TO IRRELEVANT---")
        return "irrelevant"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    logging.info("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logging.info(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        logging.info("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    logging.info("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 2)

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        logging.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        logging.info("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            logging.info("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            logging.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            logging.info("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        logging.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        logging.info("---DECISION: MAX RETRIES REACHED---")
        return "max retries"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("irrelevant", handle_irrelevant)  # handle irrelevant
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
        "irrelevant": "irrelevant",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Set irrelevant as end node
workflow.add_edge("irrelevant", END)

# Compile
graph = workflow.compile()
