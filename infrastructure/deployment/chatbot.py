import logging
import operator
import os
import traceback
from functools import wraps
from typing import Annotated
from typing import Any
from typing import Dict
from typing import List

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langgraph.constants import END
from langgraph.constants import Send
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing_extensions import Literal
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### LLM
local_llm = str(os.getenv("LLM_MODEL"))
llm = ChatOllama(model=local_llm, temperature=0.0, base_url=os.getenv("OLLAMA_URI"))
embedder = OllamaEmbeddings(
    model=str(os.getenv("EMBEDDING_MODEL")), base_url=os.getenv("OLLAMA_URI")
)

### Retriever
client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST"), port=int(os.getenv("CHROMA_PORT"))
)
vectorstore = Chroma(
    collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
    client=client,
    embedding_function=embedder,
    create_collection_if_not_exists=False,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

## Components

### Improve Query
improve_query_instructions = """
You are tasked with improving the user's query to make it more relevant to Rummikub.

Guidelines:
1. **Question Analysis:**
   - **Do not force connections**: Avoid making assumptions or altering the query that does not align with the context.
   - **Do not make assumptions**: Never make assumptions about the question.
   - **Do not alter the user's intent**: Maintain the original intent of the question.

2. **Query Improvement:**
   - Focus on game elements such as rules, gameplay, sets, runs, scoring, etc.
   - Maintain original context and intent
   - Add relevant Rummikub terminology if necessary (e.g., tiles, sets, runs).
   - You can use the previous context to understand the user's intent better.
"""

improve_query_prompt = """
Instructions:
{improve_query_instructions}

Previous context:
{formatted_context}

Latest question:
{latest_question}

Return JSON: ("improved_query": "<query>")
"""

### Router

router_instructions = """
You are a question router that classifies the user question into appropriate data sources based on their relevance to Rummikub.

It is important that you only return route information about the user's question and not the previous messages.

### Guidelines:
1. **Primary Task:**
   - Determine whether the user's question is about Rummikub. Questions should be about its rules, gameplay, strategy, tiles, or mechanics.
   - Analyze the previous messages and the user's question to make an informed decision. The user's question can be a follow-up or a new query.
   - Avoid assuming connections between unrelated topics and Rummikub unless explicitly stated by the user.

2. **Use Context Selectively:**
   - Only refer to the provided previous messages if the question is ambiguous and the context is necessary to make a decision.
   - Do not infer a connection between the question and Rummikub if it mentions unrelated terms, games, or concepts.
   - Use the context about Rummikub to understand a small bit about the game and its rules.

3. **Return:**
It is important to not include any other information in the response other than the structured output.
Call the RoutingInformation tool for the structured output so that the output is in the correct format.
"""

router_prompt = """
Analyze the user’s question and determine the most appropriate data source.
Look at the previous messages and the user's question to make an informed decision. If the question is connected to the previous messages which makes it clear that the question is about Rummikub, then route it either to the vectorstore or websearch depending on the context.
Do not force a connection between the question and Rummikub if it is not related. Do not make assumptions or reinterpret the question to fit Rummikub.
It is important that you make one routing decision of the users question and not the previous messages.

#### Context About Rummikub:
Rummikub is a **tile-based** game where players aim to be the first to empty their rack by forming valid sets or runs of numbered tiles. The game involves two types of valid combinations: sets (groups of the same number in different colors) and runs (consecutive numbers in the same color). Jokers can substitute for any tile. Questions about other games or concepts are unrelated.

#### User’s Question:
{question}

#### Previous Messages:
{formatted_context}

### Task:
1. Decide the appropriate <datasource>:
   - **`vectorstore`:** For questions explicitly about Rummikub.
   - **`websearch`:** For questions about recent events or updates related to Rummikub.
   - **`irrelevant`:** For all other questions, including those about unrelated topics.

2. Provide <reasoning> for your choice:
   - Explain briefly why the question fits the selected category.
   - For unclear questions, explain why the context provided does or does not clarify the intent.
"""

### Retrieval Grader

doc_grader_instructions = """
You are an evaluator tasked with determining the relevance of a retrieved document to a user's question.

Use the following **criteria for relevance**:
1. **Direct Match**: The document directly addresses the user's question by answering it or discussing its main subject.
2. **Contextual Help**: Even if the document does not provide a direct answer, it could offer partial information, context, or clues that contribute to answering the question. This is considered relevant.
3. **Alignment**: The document must contain specific keywords, synonyms, phrases, or related ideas that align closely with the user's question.

**Important Considerations**:
- Be **strict but fair**: Grade as 'not relevant' only if the document provides no meaningful connection to the user's question.
- Avoid being overly conservative: Even partial or incomplete information should be considered relevant if it adds value to the answer.
- Use an **objective lens**: Base your decision solely on the content of the document and its connection to the question, without assumptions.

Your evaluation must balance strictness with fairness to ensure that useful documents are not unnecessarily excluded.
"""

doc_grader_prompt = """
Here is a retrieved document:

{document}

Here is the user's question:

{improved_query}

Using the criteria provided:
1. Determine whether the document is relevant to the question.
2. Base your assessment on whether the document directly or indirectly addresses the question, provides partial information, or offers contextual relevance. These are all considered relevant.

Return JSON with a single key, `binary_score`, as either 'yes' (relevant) or 'no' (not relevant). Make your assessment objective and based solely on the content of the document and its alignment with the question.
"""

### Generate

rag_instructions = """
You are an assistant tasked with answering questions about Rummikub based on provided context. Your response must be well-structured, professional, and formatted for clear readability.

It is important that you never return the question or the improved question in the response.

### Evaluation Criteria for Responses
1. **Accuracy**: Your answer must be directly supported by the provided context.
2. **Clarity**: Provide straightforward, clear, and concise answers. Avoid redundant phrases or ambiguous phrasing.
3. **Completeness**: Fully address the users question based on the available context and improved query. If the context is insufficient, indicate this and offer actionable suggestions.
4. **Relevance**: Stick strictly to the information derived from the provided context. Avoid unnecessary repetition of the query or context unless essential for clarity.
5. **Presentation**: Avoid unnecessary headers like **Answer** or extraneous symbols such as `/n`. Format the response in a clean, natural style.
6. **Professionalism**: Never return the question or the improved question in the response.

### Best Practices
- **Stay Grounded**: Only provide information found in the context.
- **Address Insufficient Context**: If the context is incomplete or missing, acknowledge this and offer guidance on what additional details might help.
- **Direct and User-Friendly**: Structure your response to directly address the query without redundant elaboration. Use a tone that is helpful and professional.
- **Answer Structure**: Present the response in a logical order, ensuring that each point is clear and directly relevant to the query.
- **Clean Formatting**: Ensure the output is formatted as a single, clean paragraph or structured list when appropriate.
"""

rag_prompt = """
You are an assistant tasked with answering questions about Rummikub based on the provided context.

### Context
{context}

### User Question
{question}

### User Improved Question
{improved_query}

### Task
1. **Analyze the Users question**: Carefully read and understand the user's question.
2. **Review the Context**: Carefully analyze the provided context and improved question to extract the most relevant information for answering the question.
3. **Handle Insufficient Context**: If the context does not provide sufficient information to answer the question:
   - Acknowledge the insufficiency explicitly.
   - Suggest additional clarifications or details needed to proceed.
4. **Answer Clearly**: Write a concise, direct, and user-friendly response. Avoid any unnecessary elaboration or repetition.
5. **Formatting**:
   - Avoid extraneous symbols, headers (like **Answer:**), or redundant breaks.
   - Present the answer as a clean paragraph or list, depending on the nature of the question.
   - Never speak as I or refer to the assistant in the response.

### Deliverable
Provide a concise, well-structured response to the query based solely on the given context.
"""

rag_prompt_after_hallucination = """
You are an assistant tasked with answering questions about Rummikub based on the provided context.

### Context
{context}

### User Question
{question}

### User Improved Question
{improved_query}

### Previous Hallucinated Response
{previous_response}

### Task
Your goal is to provide a factual and contextually accurate response, avoiding any inaccuracies or unsupported claims from the previous response.

1. **Stay Grounded**: Ensure your answer is entirely based on the provided context. Do not include details, assumptions, or fabrications not explicitly found in the context.
2. **Avoid Hallucination**: Treat the previous response only as a reference for what to avoid. Do not rephrase or reuse its content.
3. **Conciseness**: Provide a clear and direct answer. Only repeat information from the context verbatim if absolutely necessary.
4. **Formatting**: Present the response cleanly:
   - Avoid unnecessary headers, symbols, or line breaks.
   - Structure the response as a single clean paragraph or structured list.
   - Never speak as I or refer to the assistant in the response.
5. **Transparency**: If the context does not contain sufficient information:
   - Clearly state the insufficiency.
   - Explain why the context wasn’t helpful.
   - Suggest additional clarifications or details for the user to provide.

### Deliverable
Write a concise and well-structured response to the user’s query, formatted for easy readability and based solely on the context provided.
"""


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


### Hallucination Grader


hallucination_grader_instructions = """
You are a teacher evaluating a student's answer for factual correctness.

### Grading Criteria:
1. The STUDENT ANSWER must be entirely grounded in the provided FACTS.
2. The STUDENT ANSWER must not include any information that is not explicitly mentioned in the FACTS ("hallucinated" information).

### Scoring:
- A score of "yes" indicates the STUDENT ANSWER fully meets both grading criteria. This is the highest score.
- A score of "no" indicates the STUDENT ANSWER fails to meet one or both grading criteria. This is the lowest score.

### Output:
- Return the score as a JSON object with the key `binary_score`, using the value "yes" or "no".

### Additional Notes:
- If any part of the STUDENT ANSWER introduces content outside the scope of the FACTS, assign a score of "no".
- Your evaluation should focus solely on the alignment between the STUDENT ANSWER and the FACTS.
"""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n QUESTION: \n\n {question} \n\n STUDENT ANSWER: \n\n {generation}

Return JSON with one key, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS."""

### Answer Grader

answer_grader_instructions = """
You are a teacher responsible for grading a student's response to a quiz question.

### Grading Criteria:
1. The STUDENT ANSWER must directly address the QUESTION and provide a relevant response.
2. The STUDENT ANSWER can include additional information beyond what is explicitly required by the QUESTION, as long as the extra information is accurate and relevant. Including such details will not negatively affect the score.
3. The STUDENT ANSWER must not omit critical information necessary to address the QUESTION adequately.

### Scoring Guidelines:
- **"yes"**: Assign this score if the STUDENT ANSWER answers the QUESTION, regardless of whether it includes extra relevant information.
- **"no"**: Assign this score if the STUDENT ANSWER fails to address the QUESTION, is incomplete, irrelevant, or contains significant inaccuracies.

### Evaluation Process:
- Focus on whether the STUDENT ANSWER fulfills the requirements of the QUESTION.
- Do not penalize additional details unless they detract from the clarity or accuracy of the response.

### Output:
Return the score as a JSON object with the key `binary_score` and the value as "yes" or "no".

### Notes for Grading:
- If the STUDENT ANSWER contains information unrelated to the QUESTION but still answers the QUESTION effectively, it can receive a "yes".
- If the STUDENT ANSWER is unclear, ambiguous, or fails to address the QUESTION meaningfully, it must receive a "no".
"""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: \n\n {generation}

Return JSON with one key, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria."""

### Search

web_search_tool = TavilySearchResults(max_results=3)


### Exception handling


class NodeError(Exception):
    """Base exception for node-specific errors"""

    pass


class DocumentProcessingError(NodeError):
    """Raised when document processing fails"""

    pass


class LLMError(NodeError):
    """Raised when LLM operations fail"""

    pass


class WebSearchError(NodeError):
    """Raised when web search operations fail"""

    pass


class InvalidInputStateError(NodeError):
    """Raised when input state is invalid"""

    pass


def handle_node_errors(func):
    """
    Decorator for handling common node errors and providing fallback behavior.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except InvalidInputStateError as e:
            logger.error(f"Validation error in {func.__name__}: {str(e)}")
            raise NodeError(f"Invalid data structure: {str(e)}")
        except LLMError as e:
            logger.error(f"LLM error in {func.__name__}: {str(e)}")
            return handle_llm_fallback(func.__name__, **kwargs)
        except WebSearchError as e:
            logger.error(f"Web search error in {func.__name__}: {str(e)}")
            return {
                "documents": [],
                "loop_web_search": kwargs.get("state", {}).get("loop_web_search", 0)
                + 1,
            }
        except Exception as e:
            logger.error(
                f"Unexpected error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            )
            raise NodeError(f"Unexpected error: {str(e)}")

    return wrapper


def handle_llm_fallback(node_name: str, **kwargs) -> Dict[str, Any]:
    """
    Provides fallback behavior for LLM failures based on the node type.
    """
    fallbacks = {
        "improve_query": {"improved_query": "", "question": ""},
        "grade_document": {"filtered_docs": [], "grades": []},
        "generate": {
            "loop_generate": kwargs.get("state", {}).get("max_retries", 0),
            "messages": [
                AIMessage(
                    content="I apologize, but I'm having trouble processing your request. Could you please rephrase your question?"
                )
            ],
        },
    }
    return fallbacks.get(node_name, {})


### Models


class ImprovedQuery(BaseModel):
    """Model for improved query response from the LLM"""

    improved_query: str


class RouteInformation(BaseModel):
    """An answer from the router with routing information based on the user's question"""

    datasource: Literal["vectorstore", "websearch", "irrelevant"]
    reasoning: str


class Grade(BaseModel):
    """Model for grading response from the LLM"""

    binary_score: Literal["yes", "no"]


### State


class OverallState(MessagesState):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    improved_query: str
    question: str
    web_search: str
    max_retries: int
    loop_generate: int
    loop_web_search: int
    documents: List[str]
    filtered_docs: Annotated[list[str], operator.add]
    grades: Annotated[list[str], operator.add]
    hallucination_grade: bool
    answer_grade: bool
    decision: str
    route: str


class OutputRouterState(TypedDict):
    """
    Output schema for the router
    """

    question: str
    route: str


class InputRetrieverState(TypedDict):
    """
    Input schema for the retriever
    """

    improved_query: str


class InputProceedToDatasourceState(TypedDict):
    """
    Input schema for the proceed_to_datasource node
    """

    route: str


class InputGeneratorState(MessagesState):
    """
    Input schema for the generator
    """

    improved_query: str
    filtered_docs: List[str]
    loop_generate: int
    hallucination_grade: bool
    question: str


class OutputRetrieverState(TypedDict):
    """
    Output schema for the retriever
    """

    documents: List[str]


class OutputGeneratorState(MessagesState):
    """
    Output schema for the generator
    """

    loop_generate: int


class InputContinueToGradingsState(TypedDict):
    """
    Input schema for the continue_to_gradings node
    """

    documents: List[str]
    improved_query: str


class InputDocumentGraderState(TypedDict):
    """
    Input schema for the document grader
    """

    document: str
    improved_query: str


class OutputDocumentGraderState(TypedDict):
    """
    Output schema for the document grader
    """

    filtered_docs: Annotated[list[str], operator.add]
    grades: Annotated[list[str], operator.add]


class InputDocumentsGradingsState(TypedDict):
    """
    Input schema for the documents gradings node
    """

    grades: List[str]
    loop_web_search: int
    max_retries: int


class OutputDocumentsGradingsState(TypedDict):
    """
    Output schema for the documents gradings node
    """

    web_search: str
    grades: List[str]


class InputWebSearchState(TypedDict):
    """
    Input schema for the web search node
    """

    improved_query: str
    question: str
    loop_web_search: int
    filtered_docs: List[str]


class ImproveQueryState(TypedDict):
    """
    Input schema for the improvement web search node
    """

    improved_query: str
    route: str


class InputImproveQueryState(MessagesState):
    """
    Input schema for the web search node
    """

    question: str
    route: str


class OutputWebSearchState(TypedDict):
    """
    Output schema for the web search node
    """

    documents: List[str]
    loop_web_search: int
    filtered_docs: List[str]


class InputDecisionState(TypedDict):
    """
    Input schema for the decision node
    """

    web_search: str


class InputHallucinationGraderState(MessagesState):
    """
    Input schema for the hallucination grader
    """

    filtered_docs: List[str]
    question: str


class OutputHallucinationGraderState(TypedDict):
    """
    Output schema for the hallucination grader
    """

    hallucination_grade: bool


class OutputAnswerGraderState(TypedDict):
    """
    Output schema for the hallucination grader
    """

    answer_grade: bool


class InputAnswerGraderState(MessagesState):
    """
    Input schema for the answer grader
    """

    question: str


class InputGradersState(TypedDict):
    """
    Input schema for the node which checks the gradings of the hallucination and answer grader
    """

    hallucination_grade: bool
    answer_grade: bool
    max_retries: int
    loop_generate: int
    loop_web_search: int


class OutputGradersState(TypedDict):
    """
    Output schema for the node which checks the gradings of the hallucination and answer grader
    """

    decision: str


class InputResetState(TypedDict):
    """
    Input schema for the reset state node
    """

    filtered_docs: List[str]
    grades: List[str]
    documents: List[str]


class OutputResetState(TypedDict):
    """
    Output schema for the reset state node
    """

    filtered_docs: List[str]
    grades: List[str]
    documents: List[str]
    loop_generate: int
    question: str
    web_search: str
    decision: str
    loop_web_search: int
    improved_query: str
    hallucination_grade: bool
    answer_grade: bool


### Nodes


@handle_node_errors
def route_question(state: MessagesState) -> OutputRouterState:
    """
    Route question to web search or RAG or none

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call

    Raises:
        ValidationError: If input validation fails
        LLMError: If routing question fails
    """
    logger.info("---ROUTE QUESTION---")
    try:
        question = state["messages"][-1].content
        context_messages = [
            f"{msg.type}: {msg.content}" for msg in state["messages"][-5:-1]
        ]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    try:
        formatted_context = "\n".join(context_messages)
        router_llm = llm.with_structured_output(RouteInformation)
        route_question_prompt_formatted = router_prompt.format(
            question=question, formatted_context=formatted_context
        )
        route_question_response: RouteInformation = router_llm.invoke(
            [
                SystemMessage(content=router_instructions),
                HumanMessage(content=route_question_prompt_formatted),
            ]
        )

        return {"route": route_question_response.datasource, "question": question}
    except Exception as e:
        raise LLMError(f"Routing question failed: {str(e)}")


@handle_node_errors
def improve_query(state: InputImproveQueryState) -> ImproveQueryState:
    """
    Improve the query based on the question for vectorstore and web search.

    Args:
        state: Contains messages with user query history

    Returns:
        Dictionary with the improved query and original question

    Raises:
        LLMError: If query improvement fails
        ValidationError: If input validation fails
    """
    logger.info("---IMPROVE QUERY---")
    try:
        question = state["question"]

        context_messages = [
            f"{msg.type}: {msg.content}" for msg in state["messages"][-5:-1]
        ]
        route = state["route"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    try:
        formatted_context = "\n".join(context_messages)
        improve_query_prompt_formatted = improve_query_prompt.format(
            formatted_context=formatted_context,
            latest_question=question,
            improve_query_instructions=improve_query_instructions,
        )
        structured_llm = llm.with_structured_output(ImprovedQuery)
        llm_response: ImprovedQuery = structured_llm.invoke(
            [HumanMessage(content=improve_query_prompt_formatted)]
        )
        return {"improved_query": llm_response.improved_query, "route": route}
    except Exception as e:
        raise LLMError(f"Query improvement failed: {str(e)}")


@handle_node_errors
def proceed_to_datasource(state: InputProceedToDatasourceState) -> str:
    """
    Proceed to the datasource based on the route

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call

    Raises:
        ValidationError: If input validation fails
    """
    try:
        route = state["route"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    if route == "vectorstore":
        logger.info("---PROCEED TO VECTORSTORE---")
        return "vectorstore"
    elif route == "websearch":
        logger.info("---PROCEED TO WEB SEARCH---")
        return "search web"


@handle_node_errors
def retrieve(state: InputRetrieverState) -> OutputRetrieverState:
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents

    Raises:
        DocumentProcessingError: If document retrieval fails
    """
    logger.info("---RETRIEVE---")
    try:
        improved_query = state["improved_query"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    try:
        documents = retriever.invoke(improved_query)
        return {"documents": documents}
    except Exception as e:
        raise DocumentProcessingError(f"Document retrieval failed: {str(e)}")


@handle_node_errors
def continue_to_gradings(state: InputContinueToGradingsState) -> List[Send]:
    """
    Continue to grade the documents for relevance to the user's question

    Args:
        state (dict): The current graph state

    Returns:
        list: List of Send objects to grade each document

    Raises:
        ValidationError: If input validation fails
    """
    try:
        improved_query = state["improved_query"]
        documents = state["documents"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    return [
        Send(
            "grade_document",
            {"document": d, "improved_query": improved_query},
        )
        for d in documents
    ]


@handle_node_errors
def grade_document(state: InputDocumentGraderState) -> OutputDocumentGraderState:
    """
    Grades a document for relevance to the user's question.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: New keys added to the state, `filtered_docs` and `grades`.

    Raises:
        ValidationError: If input validation fails
        LLMError: If document grading fails
    """
    try:
        improved_query = state["improved_query"]
        doc = state["document"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    try:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=doc.page_content, improved_query=improved_query
        )
        document_grader_llm = llm.with_structured_output(Grade)
        result: Grade = document_grader_llm.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = result.binary_score
        if grade.lower() == "yes":
            logger.info("---GRADE: DOCUMENT RELEVANT---")
            return {"filtered_docs": [doc], "grades": [grade]}
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            return {"grades": [grade], "filtered_docs": []}
    except Exception as e:
        raise LLMError(f"Document grading failed: {str(e)}")


@handle_node_errors
def grade_docs(state: InputDocumentsGradingsState) -> OutputDocumentsGradingsState:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state

    Raises:
        ValidationError: If input validation fails
    """
    logger.info("---CHECK IF WEB SEARCH IS NECESSARY---")
    try:
        grades = state.get("grades", [])
        max_retries = state.get("max_retries", 3)
        loop_web_search = state.get("loop_web_search", 0)
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")
    if loop_web_search >= max_retries:
        logger.info("---MAX RETRIES REACHED---")
        web_search = "No"
    elif all(x.lower() == "no" for x in grades):
        logger.info("---NO RELEVANT DOCUMENTS---")
        web_search = "Yes"
    else:
        logger.info("---RELEVANT DOCUMENTS FOUND---")
        web_search = "No"
    grades.clear()
    return {"web_search": web_search, "grades": grades}


@handle_node_errors
def decide_to_generate(state: InputDecisionState) -> str:
    """
    Determines whether to generate an answer or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call

    Raises:
        ValidationError: If input validation fails
    """

    logger.info("---ASSESS GRADED DOCUMENTS---")
    try:
        web_search = state["web_search"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    if web_search == "Yes":
        logger.info("---DECISION: NO RELEVANT DOCUMENTS, INCLUDE WEB SEARCH---")
        return "search web"
    else:
        logger.info("---DECISION: GENERATE---")
        return "generate response"


@handle_node_errors
def generate(state: InputGeneratorState) -> OutputGeneratorState:
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation

    Raises:
        ValidationError: If input validation fails
        DocumentProcessingError: If document processing fails
        LLMError: If generation fails
    """
    try:
        improved_query = state["improved_query"]
        documents = state["filtered_docs"]
        halucination_grade = state.get("hallucination_grade", False)
        loop_generate = state.get("loop_generate", 0)
        last_message = state["messages"][-1]
        question = state["question"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    try:
        docs_txt = format_docs(documents)
    except Exception as e:
        raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    if halucination_grade:
        logger.info("---ADJUST GENERATION AFTER HALLUCINATION---")
        try:
            rag_halucination_prompt_formatted = rag_prompt_after_hallucination.format(
                context=docs_txt,
                improved_query=improved_query,
                previous_response=last_message.content,
                question=question,
            )
            generation = llm.invoke(
                [SystemMessage(content=rag_instructions)]
                + [HumanMessage(content=rag_halucination_prompt_formatted)]
            )
            return {
                "loop_generate": loop_generate + 1,
                "messages": [AIMessage(content=generation.content, id=last_message.id)],
            }
        except Exception as e:
            raise LLMError(f"Generation after hallucination failed: {str(e)}")

    logger.info("---GENERATE---")
    try:
        rag_prompt_formatted = rag_prompt.format(
            context=docs_txt, improved_query=improved_query, question=question
        )
        generation = llm.invoke(
            [SystemMessage(content=rag_instructions)]
            + [HumanMessage(content=rag_prompt_formatted)]
        )
        return {
            "loop_generate": loop_generate + 1,
            "messages": [AIMessage(content=generation.content)],
        }
    except Exception as e:
        raise LLMError(f"Generation failed: {str(e)}")


@handle_node_errors
def search_web(state: InputWebSearchState) -> OutputWebSearchState:
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents

    Raises:
        ValidationError: If input validation fails
        WebSearchError: If web search fails
    """

    logger.info("---WEB SEARCH---")
    try:
        query = state["improved_query"]
        loop_web_search = state.get("loop_web_search", 0)
        filtered_docs = state.get("filtered_docs", [])
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")
    try:
        docs = web_search_tool.invoke({"query": query})
        web_results = "\n".join([d["content"] for d in docs])
        documents = [Document(page_content=web_results)]
        filtered_docs.clear()
        return {
            "documents": documents,
            "loop_web_search": loop_web_search + 1,
            "filtered_docs": filtered_docs,
        }
    except Exception as e:
        raise WebSearchError(f"Web search failed: {str(e)}")


@handle_node_errors
def grade_hallucination(
    state: InputHallucinationGraderState,
) -> OutputHallucinationGraderState:
    """
    Determines whether the generation is grounded in the document

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call

    Raises:
        ValidationError: If input validation fails
        LLMError: If hallucination grading fails
    """
    logger.info("---GRADE HALLUCINATION---")
    try:
        documents = state["filtered_docs"]
        generation = state["messages"][-1]
        question = state["question"]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")
    try:
        hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
            documents=format_docs(documents),
            generation=generation.content,
            question=question,
        )
        hallucination_llm = llm.with_structured_output(Grade)
        result: Grade = hallucination_llm.invoke(
            [SystemMessage(content=hallucination_grader_instructions)]
            + [HumanMessage(content=hallucination_grader_prompt_formatted)]
        )
        grade = result.binary_score
        if grade.lower() == "yes":
            logger.info("---DECISION: GENERATION GROUNDED IN DOCUMENT---")
            return {"hallucination_grade": False}
        return {"hallucination_grade": True}
    except Exception as e:
        raise LLMError(f"Hallucination grading failed: {str(e)}")


@handle_node_errors
def grade_answer(state: InputAnswerGraderState) -> OutputAnswerGraderState:
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call

    Raises:
        ValidationError: If input validation fails
    """

    logger.info("---GRADE GENERATION vs QUESTION---")
    try:
        question = state["question"]
        generation = state["messages"][-1]
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")
    try:
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        answer_llm = llm.with_structured_output(Grade)
        result: Grade = answer_llm.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = result.binary_score
        if grade.lower() == "yes":
            logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
            return {"answer_grade": False}
        return {"answer_grade": True}
    except Exception as e:
        raise LLMError(f"Answer grading failed: {str(e)}")


@handle_node_errors
def grade_generation_v_documents_and_question(
    state: InputGradersState,
) -> OutputGradersState:
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call

    Raises:
        ValidationError: If input validation fails
    """

    logger.info("---CHECK GRADINGS---")
    try:
        hallucination_grade = state.get("hallucination_grade", False)
        answer_grade = state.get("answer_grade", False)
        max_retries = state.get("max_retries", 3)
        loop_generate = state.get("loop_generate", 0)
        loo_web_search = state.get("loop_web_search", 0)
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")

    if not hallucination_grade and not answer_grade:
        logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
        decision = "useful"
    elif loop_generate >= max_retries or loo_web_search >= max_retries:
        logger.info("---DECISION: MAX RETRIES REACHED---")
        decision = "max retries"
    elif answer_grade:
        logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        decision = "not useful"
    elif hallucination_grade:
        logger.info("---DECISION: GENERATION HALLUCINATED---")
        decision = "not supported"
    else:
        logger.info("---DECISION: GENERATION NOT GRADED---")
        decision = "not supported"
    return {"decision": decision}


def handle_irrelevant(state) -> MessagesState:
    """
    Handle irrelevant questions
    :param state: The current graph state
    :return: New message to add to the state
    """
    logger.info("---HANDLE IRRELEVANT---")
    irrelevant_message = AIMessage(
        content="The question is either not clear or not relevant to Rummikub. I can answer any questions related to the game Rummikub."
    )
    return {"messages": [irrelevant_message]}


@handle_node_errors
def reset_state(state: InputResetState) -> OutputResetState:
    """
    Reset the state while keeping messages intact.

    Args:
        state: The current state object

    Returns:
        dict: The reset state object

    Raises:
        ValidationError: If input validation fails
    """
    logger.info("---RESET STATE---")
    try:
        filtered_docs = state.get("filtered_docs", [])
        grades = state.get("grades", [])
        documents = state.get("documents", [])
        documents.clear()
        filtered_docs.clear()
        grades.clear()
    except (KeyError, IndexError) as e:
        raise InvalidInputStateError(f"Invalid input state: {str(e)}")
    return {
        "filtered_docs": filtered_docs,
        "grades": grades,
        "documents": documents,
        "loop_generate": 0,
        "question": "",
        "web_search": "",
        "decision": "",
        "loop_web_search": 0,
        "improved_query": "",
        "hallucination_grade": False,
        "answer_grade": False,
    }


### Graph
workflow = StateGraph(OverallState, input=MessagesState, output=MessagesState)

workflow.add_node("improve_query", improve_query)
workflow.add_node("route_question", route_question)
workflow.add_node("websearch", search_web)
workflow.add_node("retrieve", retrieve)
workflow.add_node("irrelevant", handle_irrelevant)
workflow.add_node("grade_document", grade_document)
workflow.add_node("grade_docs", grade_docs)
workflow.add_node("generate", generate)
workflow.add_node("grade_hallucination", grade_hallucination)
workflow.add_node("grade_answer", grade_answer)
workflow.add_node(
    "grade_generation_v_documents_and_question",
    grade_generation_v_documents_and_question,
)
workflow.add_node("reset_state", reset_state)

workflow.set_entry_point("route_question")
workflow.add_conditional_edges(
    "route_question",
    lambda x: x["route"],
    {
        "vectorstore": "improve_query",
        "websearch": "improve_query",
        "irrelevant": "irrelevant",
    },
)
workflow.add_conditional_edges(
    "improve_query",
    proceed_to_datasource,
    {"search web": "websearch", "vectorstore": "retrieve"},
)
workflow.add_conditional_edges("websearch", continue_to_gradings, ["grade_document"])
workflow.add_conditional_edges("retrieve", continue_to_gradings, ["grade_document"])
workflow.add_edge("grade_document", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    decide_to_generate,
    {
        "search web": "websearch",
        "generate response": "generate",
    },
)
workflow.add_edge("generate", "grade_hallucination")
workflow.add_edge("generate", "grade_answer")
workflow.add_edge("grade_hallucination", "grade_generation_v_documents_and_question")
workflow.add_edge("grade_answer", "grade_generation_v_documents_and_question")
workflow.add_conditional_edges(
    "grade_generation_v_documents_and_question",
    lambda x: x["decision"],
    {
        "not supported": "generate",
        "useful": "reset_state",
        "not useful": "websearch",
        "max retries": "reset_state",
    },
)
workflow.add_edge("irrelevant", "reset_state")
workflow.add_edge("reset_state", END)

graph = workflow.compile()
