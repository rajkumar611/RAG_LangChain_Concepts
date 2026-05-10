from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

LC_SESSIONS: dict[str, list] = {}
_lc_vectorstore = None

class LCPromptReq(BaseModel):
    role: str
    question: str

class LCTextReq(BaseModel):
    text: str

class LCQReq(BaseModel):
    question: str

class LCTopicReq(BaseModel):
    topic: str

class LCMemReq(BaseModel):
    message: str
    session_id: str


# ── 1. Prompt Management ──────────────────────────────────────────────────────
@router.post("/langchain/prompt")
def lc_prompt(req: LCPromptReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    lc = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}. Answer in under 2 sentences."),
        ("human", "{question}"),
    ])
    answer = (template | lc | StrOutputParser()).invoke({"role": req.role, "question": req.question})
    return {
        "rendered": f"[system] You are a {req.role}. Answer in under 2 sentences.\n[human] {req.question}",
        "answer": answer,
    }


# ── 2. LLM Chaining ──────────────────────────────────────────────────────────
@router.post("/langchain/chaining")
def lc_chaining(req: LCTextReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    lc = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
    p  = StrOutputParser()
    step1 = (ChatPromptTemplate.from_template(
        "Translate to English. Return ONLY the translation.\n\n{text}") | lc | p).invoke({"text": req.text})
    step2 = (ChatPromptTemplate.from_template(
        "Summarise in exactly one sentence:\n\n{t}") | lc | p).invoke({"t": step1})
    step3 = (ChatPromptTemplate.from_template(
        'Wrap in JSON: {{"summary": "..."}}\nSummary: {s}') | lc | p).invoke({"s": step2})
    return {"steps": [
        {"label": "1 · Translate", "output": step1},
        {"label": "2 · Summarise", "output": step2},
        {"label": "3 · JSON",      "output": step3},
    ]}


# ── 3. RAG (fixed knowledge base) ────────────────────────────────────────────
def _get_lc_vs():
    global _lc_vectorstore
    if _lc_vectorstore:
        return _lc_vectorstore
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    docs = [
        Document(page_content="Python is a high-level programming language prized for its readability."),
        Document(page_content="LangChain is a framework for building LLM-powered applications with chains and agents."),
        Document(page_content="Vector databases store embeddings and enable fast semantic search."),
        Document(page_content="RAG stands for Retrieval Augmented Generation — fetch relevant context, then generate."),
        Document(page_content="FAISS is Meta AI's library for fast similarity search over dense vectors."),
        Document(page_content="Embeddings convert text to numerical vectors that capture semantic meaning."),
        Document(page_content="Chunking splits large documents into smaller pieces before embedding."),
    ]
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _lc_vectorstore = FAISS.from_documents(docs, emb)
    return _lc_vectorstore

@router.post("/langchain/rag")
def lc_rag(req: LCQReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    vs       = _get_lc_vs()
    chunks   = vs.as_retriever(search_kwargs={"k": 2}).invoke(req.question)
    context  = "\n".join(f"- {d.page_content}" for d in chunks)
    lc       = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
    answer   = (ChatPromptTemplate.from_template(
        "Use ONLY the context to answer.\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nOne sentence answer:")
        | lc | StrOutputParser()).invoke({"ctx": context, "q": req.question})
    return {"chunks": [d.page_content for d in chunks], "answer": answer}


# ── 4. Memory ─────────────────────────────────────────────────────────────────
@router.post("/langchain/memory")
def lc_memory(req: LCMemReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage
    if req.session_id not in LC_SESSIONS:
        LC_SESSIONS[req.session_id] = []
    history = LC_SESSIONS[req.session_id]
    lc      = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
    prompt  = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant. Remember everything the user tells you."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    answer = (prompt | lc | StrOutputParser()).invoke({"history": history, "input": req.message})
    history.append(HumanMessage(content=req.message))
    history.append(AIMessage(content=answer))
    return {
        "answer": answer,
        "history": [{"role": "user" if isinstance(m, HumanMessage) else "bot", "text": m.content}
                    for m in history],
    }

@router.delete("/langchain/memory/{session_id}")
def lc_memory_clear(session_id: str):
    LC_SESSIONS.pop(session_id, None)
    return {"cleared": True}


# ── 5. Tools & Function Calling ───────────────────────────────────────────────
@router.post("/langchain/tools")
def lc_tools(req: LCQReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a maths expression, e.g. '25 * 4 + 10'."""
        try:    return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e: return f"Error: {e}"

    @tool
    def get_weather(city: str) -> str:
        """Return current weather for a city (mock data)."""
        data = {"london": "15°C, Cloudy", "singapore": "32°C, Humid",
                "new york": "22°C, Sunny", "sydney": "19°C, Partly Cloudy"}
        return data.get(city.lower(), "No weather data for that city.")

    @tool
    def word_count(text: str) -> str:
        """Count the number of words in a piece of text."""
        return str(len(text.split()))

    tools     = [calculator, get_weather, word_count]
    tools_map = {t.name: t for t in tools}
    lc        = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
    resp      = lc.bind_tools(tools).invoke(req.question)

    content    = resp.content
    tool_calls = []

    if isinstance(content, str):
        return {"tool_calls": [], "answer": content}

    def _get(block, key):
        return block[key] if isinstance(block, dict) else getattr(block, key, None)

    for block in content:
        if _get(block, "type") == "tool_use":
            name   = _get(block, "name")
            inputs = _get(block, "input")
            result = tools_map[name].invoke(inputs)
            tool_calls.append({"tool": name, "input": inputs, "result": str(result)})

    if tool_calls:
        ctx    = "\n".join(f"{tc['tool']}({tc['input']}) = {tc['result']}" for tc in tool_calls)
        answer = (ChatPromptTemplate.from_template(
            "Question: {q}\n\nTool results:\n{ctx}\n\nAnswer using these results:")
            | lc | StrOutputParser()).invoke({"q": req.question, "ctx": ctx})
    else:
        texts  = [_get(b, "text") for b in content if _get(b, "type") == "text"]
        answer = " ".join(t for t in texts if t) or "No answer."

    return {"tool_calls": tool_calls, "answer": answer}


# ── 6. Document Processing ────────────────────────────────────────────────────
@router.post("/langchain/documents")
def lc_documents(req: LCTextReq):
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    doc  = Document(page_content=req.text)
    cs   = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n\n")
    rs   = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    return {
        "total_chars": len(req.text),
        "char_chunks": [{"i": i+1, "chars": len(c.page_content), "text": c.page_content}
                        for i, c in enumerate(cs.split_documents([doc]))],
        "rec_chunks":  [{"i": i+1, "chars": len(c.page_content), "text": c.page_content}
                        for i, c in enumerate(rs.split_documents([doc]))],
    }


# ── 7. Output Parsers ─────────────────────────────────────────────────────────
@router.post("/langchain/parsers")
def lc_parsers(req: LCTopicReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
    lc  = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
    lp  = CommaSeparatedListOutputParser()
    jp  = JsonOutputParser()
    lst = (ChatPromptTemplate.from_template(
        "List five items related to '{t}'. {fi}").partial(fi=lp.get_format_instructions())
        | lc | lp).invoke({"t": req.topic})
    jsn = (ChatPromptTemplate.from_template(
        "Return JSON about '{t}' with keys: name, description, key_facts (array of 3). {fi}")
        .partial(fi=jp.get_format_instructions()) | lc | jp).invoke({"t": req.topic})
    txt = (ChatPromptTemplate.from_template(
        "Write one sentence explaining '{t}'.") | lc | StrOutputParser()).invoke({"t": req.topic})
    return {"string_output": txt, "list_output": lst, "json_output": jsn}


# ── 8. Single Agent ───────────────────────────────────────────────────────────
@router.post("/langchain/agent")
def lc_agent_ep(req: LCQReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.prebuilt import create_react_agent

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a maths expression, e.g. '100 * 1.35'."""
        try:    return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e: return f"Error: {e}"

    @tool
    def get_exchange_rate(currency_pair: str) -> str:
        """Get exchange rate between two currencies. Format: 'USD_SGD'."""
        rates = {"usd_sgd": 1.35, "eur_usd": 1.08, "gbp_usd": 1.27, "inr_usd": 0.012}
        key   = currency_pair.lower().replace("/", "_").replace("-", "_")
        parts = key.split("_")
        rate  = rates.get(key)
        return f"1 {parts[0].upper()} = {rate} {parts[1].upper()}" if rate else "Rate not found."

    @tool
    def get_country_info(country: str) -> str:
        """Return basic facts about a country. Supports: Singapore, India, USA."""
        info = {"singapore": "City-state in Southeast Asia. Population ~6M. Currency: SGD.",
                "india": "South Asian country. Population ~1.4B. Currency: INR.",
                "usa": "North American country. Population ~330M. Currency: USD."}
        return info.get(country.lower(), "Country info not available.")

    lc     = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)
    agent  = create_react_agent(lc, [calculator, get_exchange_rate, get_country_info],
                                prompt="You are a helpful assistant. Use tools when needed. Think step by step.")
    result = agent.invoke({"messages": [{"role": "user", "content": req.question}]})

    steps, pending = [], {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                pending[tc["id"]] = {"tool": tc["name"], "input": tc["args"], "result": ""}
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id in pending:
                pending[msg.tool_call_id]["result"] = msg.content
                steps.append(pending.pop(msg.tool_call_id))

    return {"steps": steps, "answer": result["messages"][-1].content}


# ── 9. Multi-Agent Simple ─────────────────────────────────────────────────────
@router.post("/langchain/multiagent")
def lc_multiagent(req: LCTopicReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
    lc = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

    research = lc.invoke([
        SystemMessage("You are a research assistant. Given a topic, return 5 concise bullet-point facts."),
        HumanMessage(req.topic),
    ]).content

    blog = lc.invoke([
        SystemMessage("You are a blog writer. Write a short 3-paragraph blog post (max 150 words) from the research notes provided. Use a friendly tone."),
        HumanMessage(f"Research notes:\n{research}"),
    ]).content

    return {"research": research, "blog": blog}


# ── 10. LangGraph Multi-Agent ─────────────────────────────────────────────────
@router.post("/langchain/langgraph")
def lc_langgraph(req: LCTopicReq):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Literal

    lc = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
    p  = StrOutputParser()
    log: list[dict] = []

    class BlogState(TypedDict):
        topic: str; research: str; draft: str
        feedback: str; final: str; revisions: int

    def manager_node(s: BlogState) -> BlogState:
        log.append({"node": "Manager", "status": "ok",
                    "detail": f"Planning pipeline for: {s['topic']}"})
        return s

    def research_node(s: BlogState) -> BlogState:
        out = (ChatPromptTemplate.from_template(
            "Research '{t}'. Return 5 concise bullet-point facts.") | lc | p).invoke({"t": s["topic"]})
        log.append({"node": "Research Agent", "status": "ok", "detail": out})
        return {"research": out}

    def writer_node(s: BlogState) -> BlogState:
        fb  = f"\n\nFeedback:\n{s['feedback']}" if s.get("feedback") else ""
        out = (ChatPromptTemplate.from_template(
            "Write a 3-paragraph blog post (max 120 words).\n\nResearch:\n{r}{fb}") | lc | p
               ).invoke({"r": s["research"], "fb": fb})
        log.append({"node": "Writer Agent", "status": "ok", "detail": out})
        return {"draft": out, "revisions": s.get("revisions", 0)}

    def reviewer_node(s: BlogState) -> BlogState:
        review = (ChatPromptTemplate.from_template(
            "Review this blog. Reply APPROVED or REVISE: <feedback>\n\nDraft:\n{d}") | lc | p
                  ).invoke({"d": s["draft"]})
        approved = review.strip().upper().startswith("APPROVED")
        log.append({"node": "Reviewer Agent",
                    "status": "approved" if approved else "revise",
                    "detail": review})
        if approved:
            return {"final": s["draft"], "feedback": ""}
        return {"feedback": review, "revisions": s.get("revisions", 0) + 1}

    def should_revise(s: BlogState) -> Literal["writer", "end"]:
        if s.get("final") or s.get("revisions", 0) >= 2: return "end"
        return "writer"

    g = StateGraph(BlogState)
    for name, fn in [("manager", manager_node), ("research", research_node),
                     ("writer",  writer_node),   ("reviewer", reviewer_node)]:
        g.add_node(name, fn)
    g.set_entry_point("manager")
    g.add_edge("manager", "research")
    g.add_edge("research", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", should_revise, {"writer": "writer", "end": END})
    lg_app = g.compile()

    result = lg_app.invoke({"topic": req.topic, "research": "", "draft": "",
                            "feedback": "", "final": "", "revisions": 0})
    return {"log": log, "final": result.get("final") or result.get("draft"),
            "revisions": result.get("revisions", 0)}
