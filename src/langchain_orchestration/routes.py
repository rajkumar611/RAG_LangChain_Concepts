import ast
import logging
import operator as op
import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
from src.config import settings  # noqa: E402

LC_MODEL = settings.haiku_model
EMB_MODEL = settings.embedding_model

# ── Session state (cleared on server restart) ─────────────────────────────────
LC_SESSIONS: dict[str, list] = {}
_lc_vectorstore = None


# ── Request models ────────────────────────────────────────────────────────────
class PromptRequest(BaseModel):
    role: str = Field(..., min_length=1, max_length=200)
    question: str = Field(..., min_length=1, max_length=2000)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)


class MemoryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


# ── Safe math evaluator (used by calculator tool) ─────────────────────────────
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval_math(node: ast.expr) -> float:
    """Recursively evaluate an AST node using only whitelisted numeric operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_math(node.left), _safe_eval_math(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_math(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


def _make_calculator_tool():
    """Return a LangChain calculator tool with safe AST-based math evaluation."""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a maths expression, e.g. '25 * 4 + 10'."""
        if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\^]+", expression):
            return "Error: only numeric expressions are supported."
        try:
            tree = ast.parse(expression, mode="eval")
            return str(round(_safe_eval_math(tree.body), 10))
        except Exception as e:
            return f"Error: {e}"

    return calculator


# ── 1. Prompt Management ──────────────────────────────────────────────────────
@router.post("/langchain/prompt")
def lc_prompt(req: PromptRequest):
    """Demonstrate LangChain prompt templates and StrOutputParser.

    Shows how ChatPromptTemplate composes a system + human message pair,
    and how piping through StrOutputParser extracts the string response.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a {role}. Answer in under 2 sentences."),
                ("human", "{question}"),
            ]
        )
        answer = (template | chat_model | StrOutputParser()).invoke(
            {"role": req.role, "question": req.question}
        )
        return {
            "rendered": f"[system] You are a {req.role}. Answer in under 2 sentences.\n[human] {req.question}",
            "answer": answer,
        }
    except Exception as e:
        logger.error("lc_prompt error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 2. LLM Chaining ──────────────────────────────────────────────────────────
@router.post("/langchain/chaining")
def lc_chaining(req: TextRequest):
    """Demonstrate sequential LangChain chaining: translate → summarise → JSON.

    Three independent chains run in sequence, each consuming the previous
    step's output. Shows how the pipe operator (|) composes LCEL chains.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        str_parser = StrOutputParser()

        step1 = (
            ChatPromptTemplate.from_template(
                "Translate to English. Return ONLY the translation.\n\n{text}"
            )
            | chat_model
            | str_parser
        ).invoke({"text": req.text})

        step2 = (
            ChatPromptTemplate.from_template("Summarise in exactly one sentence:\n\n{t}")
            | chat_model
            | str_parser
        ).invoke({"t": step1})

        step3 = (
            ChatPromptTemplate.from_template('Wrap in JSON: {{"summary": "..."}}\nSummary: {s}')
            | chat_model
            | str_parser
        ).invoke({"s": step2})

        return {
            "steps": [
                {"label": "1 · Translate", "output": step1},
                {"label": "2 · Summarise", "output": step2},
                {"label": "3 · JSON", "output": step3},
            ]
        }
    except Exception as e:
        logger.error("lc_chaining error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 3. RAG (fixed knowledge base) ────────────────────────────────────────────
def _get_lc_vectorstore():
    """Lazy-load a small fixed FAISS vectorstore with 7 AI/ML facts.

    This is a self-contained demonstration store, separate from the uploaded
    document corpus used by the RAG module.
    """
    global _lc_vectorstore
    if _lc_vectorstore:
        return _lc_vectorstore

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Python is a high-level programming language prized for its readability."
        ),
        Document(
            page_content="LangChain is a framework for building LLM-powered applications with chains and agents."
        ),
        Document(page_content="Vector databases store embeddings and enable fast semantic search."),
        Document(
            page_content="RAG stands for Retrieval Augmented Generation — fetch relevant context, then generate."
        ),
        Document(
            page_content="FAISS is Meta AI's library for fast similarity search over dense vectors."
        ),
        Document(
            page_content="Embeddings convert text to numerical vectors that capture semantic meaning."
        ),
        Document(
            page_content="Chunking splits large documents into smaller pieces before embedding."
        ),
    ]
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    _lc_vectorstore = FAISS.from_documents(docs, emb)
    return _lc_vectorstore


@router.post("/langchain/rag")
def lc_rag(req: QuestionRequest):
    """Demonstrate LangChain RAG with a fixed in-memory FAISS vectorstore.

    Uses a hardcoded 7-document AI/ML knowledge base. Shows how LangChain's
    retriever interface abstracts the underlying vector store.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        vectorstore = _get_lc_vectorstore()
        chunks = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(req.question)
        context = "\n".join(f"- {d.page_content}" for d in chunks)
        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        answer = (
            ChatPromptTemplate.from_template(
                "Use ONLY the context to answer.\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nOne sentence answer:"
            )
            | chat_model
            | StrOutputParser()
        ).invoke({"ctx": context, "q": req.question})
        return {"chunks": [d.page_content for d in chunks], "answer": answer}
    except Exception as e:
        logger.error("lc_rag error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 4. Memory ─────────────────────────────────────────────────────────────────
@router.post("/langchain/memory")
def lc_memory(req: MemoryRequest):
    """Demonstrate per-session conversation memory via MessagesPlaceholder.

    Each session_id maintains its own history list. Messages are stored as
    LangChain HumanMessage/AIMessage objects and injected into each prompt.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        if req.session_id not in LC_SESSIONS:
            LC_SESSIONS[req.session_id] = []
        history = LC_SESSIONS[req.session_id]

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=256)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a friendly assistant. Remember everything the user tells you."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        answer = (prompt | chat_model | StrOutputParser()).invoke(
            {"history": history, "input": req.message}
        )
        history.append(HumanMessage(content=req.message))
        history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "history": [
                {"role": "user" if isinstance(m, HumanMessage) else "bot", "text": m.content}
                for m in history
            ],
        }
    except Exception as e:
        logger.error("lc_memory error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.delete("/langchain/memory/{session_id}")
def lc_memory_clear(session_id: str):
    """Clear all conversation history for a given session."""
    LC_SESSIONS.pop(session_id, None)
    return {"cleared": True}


# ── 5. Tools & Function Calling ───────────────────────────────────────────────
@router.post("/langchain/tools")
def lc_tools(req: QuestionRequest):
    """Demonstrate LangChain tool binding with calculator, weather, and word-count tools.

    Shows how bind_tools registers Python functions as Claude tools, how the
    model decides which tool to call, and how results are fed back for a
    final synthesized answer.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import tool

        calculator = _make_calculator_tool()

        @tool
        def get_weather(city: str) -> str:
            """Return current weather for a city (mock data)."""
            data = {
                "london": "15°C, Cloudy",
                "singapore": "32°C, Humid",
                "new york": "22°C, Sunny",
                "sydney": "19°C, Partly Cloudy",
            }
            return data.get(city.lower(), "No weather data for that city.")

        @tool
        def word_count(text: str) -> str:
            """Count the number of words in a piece of text."""
            return str(len(text.split()))

        tools = [calculator, get_weather, word_count]
        tools_map = {t.name: t for t in tools}
        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=512)
        resp = chat_model.bind_tools(tools).invoke(req.question)
        content = resp.content

        if isinstance(content, str):
            return {"tool_calls": [], "answer": content}

        def _get_field(block, key):
            return block[key] if isinstance(block, dict) else getattr(block, key, None)

        tool_calls = []
        for block in content:
            if _get_field(block, "type") == "tool_use":
                name = _get_field(block, "name")
                inputs = _get_field(block, "input")
                result = tools_map[name].invoke(inputs)
                tool_calls.append({"tool": name, "input": inputs, "result": str(result)})

        if tool_calls:
            ctx = "\n".join(f"{tc['tool']}({tc['input']}) = {tc['result']}" for tc in tool_calls)
            answer = (
                ChatPromptTemplate.from_template(
                    "Question: {q}\n\nTool results:\n{ctx}\n\nAnswer using these results:"
                )
                | chat_model
                | StrOutputParser()
            ).invoke({"q": req.question, "ctx": ctx})
        else:
            texts = [_get_field(b, "text") for b in content if _get_field(b, "type") == "text"]
            answer = " ".join(t for t in texts if t) or "No answer."

        return {"tool_calls": tool_calls, "answer": answer}
    except Exception as e:
        logger.error("lc_tools error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 6. Document Processing ────────────────────────────────────────────────────
@router.post("/langchain/documents")
def lc_documents(req: TextRequest):
    """Demonstrate CharacterTextSplitter vs RecursiveCharacterTextSplitter.

    Both splitters use chunk_size=200 and chunk_overlap=20 so the difference
    in resulting chunks illustrates how the splitting strategy affects output.
    """
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

        doc = Document(page_content=req.text)
        char_split = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n\n")
        rec_split = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        return {
            "total_chars": len(req.text),
            "char_chunks": [
                {"i": i + 1, "chars": len(c.page_content), "text": c.page_content}
                for i, c in enumerate(char_split.split_documents([doc]))
            ],
            "rec_chunks": [
                {"i": i + 1, "chars": len(c.page_content), "text": c.page_content}
                for i, c in enumerate(rec_split.split_documents([doc]))
            ],
        }
    except Exception as e:
        logger.error("lc_documents error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 7. Output Parsers ─────────────────────────────────────────────────────────
@router.post("/langchain/parsers")
def lc_parsers(req: TopicRequest):
    """Demonstrate StrOutputParser, JsonOutputParser, and CommaSeparatedListOutputParser.

    All three parsers receive the same topic but use different format instructions,
    showing how LangChain parsers shape LLM output into typed Python objects.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import (
            CommaSeparatedListOutputParser,
            JsonOutputParser,
            StrOutputParser,
        )
        from langchain_core.prompts import ChatPromptTemplate

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=512)
        list_parser = CommaSeparatedListOutputParser()
        json_parser = JsonOutputParser()

        list_output = (
            ChatPromptTemplate.from_template("List five items related to '{t}'. {fi}").partial(
                fi=list_parser.get_format_instructions()
            )
            | chat_model
            | list_parser
        ).invoke({"t": req.topic})

        json_output = (
            ChatPromptTemplate.from_template(
                "Return JSON about '{t}' with keys: name, description, key_facts (array of 3). {fi}"
            ).partial(fi=json_parser.get_format_instructions())
            | chat_model
            | json_parser
        ).invoke({"t": req.topic})

        str_output = (
            ChatPromptTemplate.from_template("Write one sentence explaining '{t}'.")
            | chat_model
            | StrOutputParser()
        ).invoke({"t": req.topic})

        return {"string_output": str_output, "list_output": list_output, "json_output": json_output}
    except Exception as e:
        logger.error("lc_parsers error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 8. Single Agent ───────────────────────────────────────────────────────────
@router.post("/langchain/agent")
def lc_agent_ep(req: QuestionRequest):
    """Demonstrate a ReAct agent via langgraph.prebuilt.create_react_agent.

    The agent has access to calculator, exchange rate, and country info tools.
    It reasons step-by-step, deciding which tools to call before answering.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        calculator = _make_calculator_tool()

        @tool
        def get_exchange_rate(currency_pair: str) -> str:
            """Get exchange rate between two currencies. Format: 'USD_SGD'."""
            rates = {"usd_sgd": 1.35, "eur_usd": 1.08, "gbp_usd": 1.27, "inr_usd": 0.012}
            key = currency_pair.lower().replace("/", "_").replace("-", "_")
            parts = key.split("_")
            rate = rates.get(key)
            return (
                f"1 {parts[0].upper()} = {rate} {parts[1].upper()}" if rate else "Rate not found."
            )

        @tool
        def get_country_info(country: str) -> str:
            """Return basic facts about a country. Supports: Singapore, India, USA."""
            info = {
                "singapore": "City-state in Southeast Asia. Population ~6M. Currency: SGD.",
                "india": "South Asian country. Population ~1.4B. Currency: INR.",
                "usa": "North American country. Population ~330M. Currency: USD.",
            }
            return info.get(country.lower(), "Country info not available.")

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=1024)
        agent = create_react_agent(
            chat_model,
            [calculator, get_exchange_rate, get_country_info],
            prompt="You are a helpful assistant. Use tools when needed. Think step by step.",
        )
        result = agent.invoke({"messages": [{"role": "user", "content": req.question}]})

        steps: list[dict] = []
        pending: dict[str, dict] = {}
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    pending[tool_call["id"]] = {
                        "tool": tool_call["name"],
                        "input": tool_call["args"],
                        "result": "",
                    }
            elif isinstance(msg, ToolMessage) and msg.tool_call_id in pending:
                pending[msg.tool_call_id]["result"] = msg.content
                steps.append(pending.pop(msg.tool_call_id))

        return {"steps": steps, "answer": result["messages"][-1].content}
    except Exception as e:
        logger.error("lc_agent error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 9. Multi-Agent Simple ─────────────────────────────────────────────────────
@router.post("/langchain/multiagent")
def lc_multiagent(req: TopicRequest):
    """Demonstrate a simple two-agent pipeline: researcher → blog writer.

    Two sequential LLM calls with different system prompts simulate specialized
    agents handing off work. The researcher produces facts; the writer crafts prose.
    """
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=1024)

        research = chat_model.invoke(
            [
                SystemMessage(
                    "You are a research assistant. Given a topic, return 5 concise bullet-point facts."
                ),
                HumanMessage(req.topic),
            ]
        ).content

        blog = chat_model.invoke(
            [
                SystemMessage(
                    "You are a blog writer. Write a short 3-paragraph blog post (max 150 words) from the research notes provided. Use a friendly tone."
                ),
                HumanMessage(f"Research notes:\n{research}"),
            ]
        ).content

        return {"research": research, "blog": blog}
    except Exception as e:
        logger.error("lc_multiagent error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})


# ── 10. LangGraph Multi-Agent ─────────────────────────────────────────────────
@router.post("/langchain/langgraph")
def lc_langgraph(req: TopicRequest):
    """Demonstrate a LangGraph StateGraph with manager → research → writer → reviewer loop.

    Builds a directed graph with a conditional edge: the reviewer either approves
    the draft or sends it back to the writer for revision (max 2 revisions total).
    """
    try:
        from typing import Literal, TypedDict

        from langchain_anthropic import ChatAnthropic
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import END, StateGraph

        chat_model = ChatAnthropic(model=LC_MODEL, max_tokens=512)
        str_parser = StrOutputParser()
        log: list[dict] = []

        class BlogState(TypedDict):
            topic: str
            research: str
            draft: str
            feedback: str
            final: str
            revisions: int

        def manager_node(state: BlogState) -> BlogState:
            log.append(
                {
                    "node": "Manager",
                    "status": "ok",
                    "detail": f"Planning pipeline for: {state['topic']}",
                }
            )
            return state

        def research_node(state: BlogState) -> BlogState:
            output = (
                ChatPromptTemplate.from_template(
                    "Research '{t}'. Return 5 concise bullet-point facts."
                )
                | chat_model
                | str_parser
            ).invoke({"t": state["topic"]})
            log.append({"node": "Research Agent", "status": "ok", "detail": output})
            return {"research": output}

        def writer_node(state: BlogState) -> BlogState:
            feedback_block = f"\n\nFeedback:\n{state['feedback']}" if state.get("feedback") else ""
            output = (
                ChatPromptTemplate.from_template(
                    "Write a 3-paragraph blog post (max 120 words).\n\nResearch:\n{r}{fb}"
                )
                | chat_model
                | str_parser
            ).invoke({"r": state["research"], "fb": feedback_block})
            log.append({"node": "Writer Agent", "status": "ok", "detail": output})
            return {"draft": output, "revisions": state.get("revisions", 0)}

        def reviewer_node(state: BlogState) -> BlogState:
            review = (
                ChatPromptTemplate.from_template(
                    "Review this blog. Reply APPROVED or REVISE: <feedback>\n\nDraft:\n{d}"
                )
                | chat_model
                | str_parser
            ).invoke({"d": state["draft"]})
            approved = review.strip().upper().startswith("APPROVED")
            log.append(
                {
                    "node": "Reviewer Agent",
                    "status": "approved" if approved else "revise",
                    "detail": review,
                }
            )
            if approved:
                return {"final": state["draft"], "feedback": ""}
            return {"feedback": review, "revisions": state.get("revisions", 0) + 1}

        def should_revise(state: BlogState) -> Literal["writer", "end"]:
            if state.get("final") or state.get("revisions", 0) >= 2:
                return "end"
            return "writer"

        graph = StateGraph(BlogState)
        for name, fn in [
            ("manager", manager_node),
            ("research", research_node),
            ("writer", writer_node),
            ("reviewer", reviewer_node),
        ]:
            graph.add_node(name, fn)

        graph.set_entry_point("manager")
        graph.add_edge("manager", "research")
        graph.add_edge("research", "writer")
        graph.add_edge("writer", "reviewer")
        graph.add_conditional_edges("reviewer", should_revise, {"writer": "writer", "end": END})
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "topic": req.topic,
                "research": "",
                "draft": "",
                "feedback": "",
                "final": "",
                "revisions": 0,
            }
        )
        return {
            "log": log,
            "final": result.get("final") or result.get("draft"),
            "revisions": result.get("revisions", 0),
        }
    except Exception as e:
        logger.error("lc_langgraph error: %s", e)
        return JSONResponse(status_code=500, content={"detail": str(e)})
