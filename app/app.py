# ✅ Enhanced LangGraph QA Workflow for High-Recall Answers with Gemini Flash

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Literal
import os
import faiss
import logging
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# ✅ FastAPI App Initialization
app = FastAPI()

# ✅ Load Gemini Flash
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Set GOOGLE_API_KEY in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", config={"temperature": 0})

# ✅ Load and Split PDFs (Sentence-level)
loader = PyPDFDirectoryLoader("./downloaded_pdfs")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=50)
all_splits = []
for doc in documents:
    filename = doc.metadata.get("source", "unknown")
    splits = text_splitter.split_documents([doc])
    for split in splits:
        split.metadata['source'] = filename
    all_splits.extend(splits)

# ✅ Embedding Model + FAISS Index
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index = faiss.IndexFlatL2(768)  # No runtime call
vector_store = FAISS(embeddings, index, InMemoryDocstore(), {})
_ = vector_store.add_documents(all_splits)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})
retriever_tool = create_retriever_tool(retriever, "retrieve_fairview_info", "Search Fairview policy and service information")

# ✅ Define LangGraph Components
class GradeDocuments(BaseModel):
    binary_score: str

def normalize_question(state: MessagesState):
    original = state["messages"][0].content
    prompt = f"""Rephrase the user query to maximize match with internal HR policies or FAQs.
Always clarify eligibility, policies, or services.
Original: '{original}'"""
    revised = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": revised.content}]}

def generate_query_or_respond(state: MessagesState):
    system_prompt = "Always use the retriever tool to find relevant answers from internal documentation."
    return {"messages": [
        llm.bind_tools([retriever_tool]).invoke([
            {"role": "system", "content": system_prompt},
            *state["messages"]
        ])
    ]}

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = f"""Given this document content:\n\n{context}\n\nCan it help answer this question: '{question}'?
Reply strictly with 'yes' or 'no' only."""
    result = llm.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])
    answer = result.binary_score.strip().lower()
    return "generate_answer" if "yes" in answer else "rewrite_question"

def rewrite_question(state: MessagesState):
    original = state["messages"][0].content
    prompt = f"Rephrase this question to better match official policy or FAQ language:\n'{original}'"
    revised = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": revised.content}]}

def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = f"""You are a Fairview HR Support Agent.
Given this question: {question}
And this context:\n{context}\n
Answer clearly in 1–2 sentences. Do not mention 'according to the document' or the document itself."""
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# ✅ LangGraph Build
graph = StateGraph(MessagesState)
graph.add_node(normalize_question)
graph.add_node(generate_query_or_respond)
graph.add_node("retrieve", ToolNode([retriever_tool]))
graph.add_node(rewrite_question)
graph.add_node(generate_answer)

graph.add_edge(START, "normalize_question")
graph.add_edge("normalize_question", "generate_query_or_respond")
graph.add_conditional_edges("generate_query_or_respond", tools_condition, {
    "tools": "retrieve",
    END: END,
})
graph.add_conditional_edges("retrieve", grade_documents, {
    "generate_answer": "generate_answer",
    "rewrite_question": "rewrite_question",
})
graph.add_edge("rewrite_question", "generate_query_or_respond")
graph.add_edge("generate_answer", END)
workflow = graph.compile()

# ✅ Dialogflow Webhook Endpoint
class DialogflowCXInput(BaseModel):
    query: str

@app.post("/webhook")
async def dialogflow_webhook(request: Request):
    body = await request.json()
    user_query = body.get("sessionInfo", {}).get("parameters", {}).get("user_input", "") or body.get("text", "")
    messages = [{"role": "user", "content": user_query}]
    result = ""

    for chunk in workflow.stream({"messages": messages}):
        for _, update in chunk.items():
            last_msg = update["messages"][-1]
            if isinstance(last_msg, AIMessage):
                result = last_msg.content
            elif isinstance(last_msg, dict):
                result = last_msg.get("content", "")
            else:
                result = str(last_msg)

    return {
        "fulfillmentResponse": {
            "messages": [{"text": {"text": [f"kb_fmla: {result}"]}}]
        }
    }
