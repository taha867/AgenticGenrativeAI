"""
Agent node implementations: remember, planner, retriever, reasoning, tool, answer.
Use async for I/O; remember and answer use Postgres store for LTM when available.
"""
import json
import logging
import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from backend.agents.memory import LTM_SYSTEM_PROMPT_TEMPLATE, MEMORY_PROMPT, MemoryDecision
from backend.core.config import Settings, getSettings
from backend.core.constants import STM_KEEP_LAST_N
from backend.rag import formatDocs, getRetriever, runSelfRag
from backend.tools.mcp_client import getMcpTools, invokeTool

logger = logging.getLogger(__name__)

# LTM namespace: ("user", user_id, "details")
LTM_NAMESPACE_DETAILS = "details"


def _ltm_text_for_prompt(store: Any, config: RunnableConfig) -> str:
    """Load user-scoped LTM lines from Postgres store for system prompts."""
    if store is None:
        return ""
    configurable = config.get("configurable") or {}
    userId = configurable.get("user_id")
    if not userId:
        return ""
    ns = ("user", userId, LTM_NAMESPACE_DETAILS)
    items = store.search(ns)
    if not items:
        return ""
    return "\n".join((getattr(it, "value", it).get("data", "") for it in items)).strip()


def _is_ai_message(m: BaseMessage) -> bool:
    return isinstance(m, AIMessage) or getattr(m, "type", None) == "ai"


def _tool_call_ids_from_ai(ai: AIMessage) -> list[str]:
    out: list[str] = []
    for tc in getattr(ai, "tool_calls", None) or []:
        if isinstance(tc, dict):
            tid = tc.get("id") or tc.get("tool_call_id")
            if tid:
                out.append(str(tid))
    return out


def _tool_messages_cover_calls(messages: list[BaseMessage], ai_index: int, ai: AIMessage) -> bool:
    needed = set(_tool_call_ids_from_ai(ai))
    if not needed:
        return True
    j = ai_index + 1
    covered: set[str] = set()
    while j < len(messages) and isinstance(messages[j], ToolMessage):
        tid = getattr(messages[j], "tool_call_id", None) or ""
        if tid:
            covered.add(str(tid))
        j += 1
    return needed <= covered


def _find_rightmost_dangling_assistant(messages: list[BaseMessage]) -> int | None:
    """Index of rightmost AIMessage with tool_calls not fully followed by ToolMessages."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if not _is_ai_message(m):
            continue
        tcs = getattr(m, "tool_calls", None) or []
        if not tcs:
            continue
        if isinstance(m, AIMessage):
            ai = m
        else:
            continue
        if not _tool_messages_cover_calls(messages, i, ai):
            return i
    return None


async def recoverLegacyToolCallsNode(
    state: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Lift a dangling assistant tool-call turn from messages into pending_ai_message and remove it
    from messages so toolNode can append [AIMessage, ToolMessage, ...] atomically.
    """
    if state.get("pending_ai_message") is not None:
        return {}
    messages = state.get("messages") or []
    idx = _find_rightmost_dangling_assistant(list(messages))
    if idx is None:
        return {}
    ai = messages[idx]
    if not isinstance(ai, AIMessage):
        return {}
    needed = _tool_call_ids_from_ai(ai)
    got = set()
    j = idx + 1
    while j < len(messages) and isinstance(messages[j], ToolMessage):
        got.add(str(getattr(messages[j], "tool_call_id", "") or ""))
        j += 1
    logger.warning(
        "recover_legacy: lifting dangling assistant tool_calls to pending (index=%s need=%s have=%s)",
        idx,
        needed,
        sorted(got),
    )
    mid = getattr(ai, "id", None)
    if not mid:
        logger.error("recover_legacy: AIMessage missing id; cannot RemoveMessage")
        return {}
    return {"pending_ai_message": ai, "messages": [RemoveMessage(id=mid)]}


async def rememberNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    store: Any = None,
    settings: Optional[Settings] = None,
    memoryExtractor: Any = None,
) -> dict[str, Any]:
    """
    LTM: extract memories from latest user message; write is_new=True to store.
    No-op if store is None or no user_id in config.
    """
    if store is None:
        return {}
    configurable = config.get("configurable") or {}
    userId = configurable.get("user_id")
    if not userId:
        return {}
    ns = ("user", userId, LTM_NAMESPACE_DETAILS)

    settings = settings or getSettings()
    if memoryExtractor is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)
        memoryExtractor = llm.with_structured_output(MemoryDecision)

    existingItems = store.search(ns)
    existing = "\n".join(getattr(it, "value", it).get("data", "") for it in existingItems) if existingItems else "(empty)"
    messages = state.get("messages") or []
    lastContent = messages[-1].content if messages else ""
    if isinstance(lastContent, list):
        lastContent = " ".join(getattr(c, "text", str(c)) for c in lastContent)
    lastText = str(lastContent).strip()
    if not lastText:
        return {}

    decision: MemoryDecision = await memoryExtractor.ainvoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            HumanMessage(content=lastText),
        ]
    )
    if not decision.should_write:
        return {}
    for mem in decision.memories:
        if mem.is_new and mem.text.strip():
            store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})
    return {}


async def plannerNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    llm: Any = None,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """LLM plans high-level steps: retrieve, use tools, synthesize."""
    settings = settings or getSettings()
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    query = state.get("query") or (state.get("messages") or []) and (state["messages"][-1].content or "")
    if isinstance(query, list):
        query = " ".join(getattr(c, "text", str(c)) for c in query)
    prompt = f"Given the user query, output a short plan (1-3 steps). Query: {query}"
    msg = await llm.ainvoke([HumanMessage(content=prompt)])
    plan = msg.content if hasattr(msg, "content") else str(msg)
    return {"planned_steps": plan}


async def retrieverNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Run retrieval (Self-RAG) with current query; put answer and sources in state."""
    settings = settings or getSettings()
    query = state.get("query") or ""
    messages = state.get("messages") or []
    if not query and messages:
        last = messages[-1]
        query = getattr(last, "content", "") or ""
    if isinstance(query, list):
        query = " ".join(getattr(c, "text", str(c)) for c in query)
    if not query:
        return {"retrieved_docs": [], "final_answer": ""}

    configurable = config.get("configurable") or {}
    uid = configurable.get("user_id") or ""
    meta_filter: dict[str, str] | None = {"user_id": uid} if uid else None
    override_k = configurable.get("max_retrieval_docs")
    if override_k is None or (isinstance(override_k, int) and override_k <= 0):
        top_k = settings.retrieval_top_k
    else:
        top_k = int(override_k)
    retriever = getRetriever(
        settings=settings,
        topK=top_k,
        metadataFilter=meta_filter,
    )
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    sourced_pipeline = configurable.get("sourced_pipeline", True)
    disallow_direct = bool(sourced_pipeline)
    answer, sources, rewriteTries, retries, retrieval_from_documents = await runSelfRag(
        query,
        retriever,
        llm,
        settings,
        disallow_direct_generation=disallow_direct,
    )
    # Sources only list chunks Self-RAG actually used (relevant_docs). No blind top-k fallback:
    # that misled users by showing unrelated PDF snippets when the answer came from general knowledge.
    docs = [{"content": s.get("content", ""), "metadata": s.get("metadata", {})} for s in sources]
    refinement_rounds = rewriteTries + retries
    return {
        "retrieved_docs": docs,
        "final_answer": answer,
        "sources": sources,
        "retrieval_from_documents": retrieval_from_documents,
        "refinement_rounds": refinement_rounds,
    }


def _phase_instruction(phase: str) -> str:
    if phase == "mcp":
        return (
            "Phase: MCP tools only (web_search is not available in this phase).\n"
            "Use retrieved context and prior tool results for factual claims about the user's documents.\n"
            "GitHub: for repositories under a specific user, use search_repositories with GitHub search qualifiers, "
            "e.g. `user:LOGIN` plus `TOPIC` (e.g. `user:taha867 agentic` or `user:taha867 agenticai`), not generic keywords alone. "
            "For an exact owner/repo, prefer `repo:owner/name` in search when applicable. "
            "If a repo is missing, say so from tool results; do not substitute unrelated popular repos.\n"
            "Call other MCP tools when appropriate. If documents and MCP tools cannot answer, output no tool calls "
            "and a brief note; the next phase may use public web search.\n"
            "Provide a concise answer or tool calls only."
        )
    if phase == "web":
        return (
            "Phase: public web search (web_search tool only).\n"
            "Call web_search when the question needs current events, public facts, or information not in prior context.\n"
            "If web search is unnecessary, reply concisely using only prior context and tool results.\n"
            "Do not invent facts; if you lack sourced material, say you cannot provide a sourced answer."
        )
    return (
        "Prioritize the user's retrieved documents when they contain the answer. "
        "If documents do not contain the answer, use tools: call web_search for public facts or current events, "
        "or other tools when appropriate. If you use only general knowledge, say so briefly. "
        "Provide a concise final answer; use tools only when needed."
    )


async def reasoningPhaseNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    llm: Any = None,
    tools: Optional[list] = None,
    settings: Optional[Settings] = None,
    phase: str = "legacy",
    store: Any = None,
) -> dict[str, Any]:
    """
    LLM reasons over query + retrieved_docs + tool_results; may output tool_calls or go to answer.
    Staged tool calls: assistant with tool_calls is stored in pending_ai_message only until toolNode runs.
    """
    if state.get("pending_ai_message") is not None:
        logger.warning("reasoningPhaseNode(%s): pending_ai_message still set; skipping LLM", phase)
        return {}

    settings = settings or getSettings()
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    messages = list(state.get("messages") or [])
    query = state.get("query") or ""
    retrievedDocs = state.get("retrieved_docs") or []
    toolResults = state.get("tool_results") or []
    finalAnswer = (state.get("final_answer") or "").strip()

    contextParts = []
    if retrievedDocs:
        contextParts.append(
            "Retrieved context:\n"
            + "\n\n".join(d.get("content", d) if isinstance(d, dict) else str(d) for d in retrievedDocs)
        )
    if toolResults:
        contextParts.append("Tool results:\n" + "\n".join(str(t) for t in toolResults))
    contextStr = "\n\n".join(contextParts) if contextParts else "No additional context."
    if finalAnswer:
        contextStr += f"\n\n(Self-RAG answer from docs: {finalAnswer})"

    instruction = _phase_instruction(phase)
    prompt = f"User query: {query}\n\n{contextStr}\n\n{instruction}"
    llm_messages = list(messages)
    summary = (state.get("summary") or "").strip()
    if summary:
        llm_messages = [SystemMessage(content=f"Conversation summary:\n{summary}")] + llm_messages
    ltm_text = _ltm_text_for_prompt(store, config)
    if ltm_text:
        llm_messages = [
            SystemMessage(
                content=(
                    "User long-term memory (trusted; use for their name and stated preferences when relevant):\n"
                    f"{ltm_text}"
                )
            )
        ] + llm_messages

    if tools:
        llmWithTools = llm.bind_tools(tools)
        response = await llmWithTools.ainvoke(llm_messages + [HumanMessage(content=prompt)])
    else:
        response = await llm.ainvoke(llm_messages + [HumanMessage(content=prompt)])

    toolCalls = getattr(response, "tool_calls", None) or []
    if toolCalls:
        for tc in toolCalls:
            if isinstance(tc, dict):
                logger.info(
                    "reasoningPhaseNode(%s): tool_call model=%s id=%s arg_keys=%s",
                    phase,
                    tc.get("name"),
                    tc.get("id"),
                    list((tc.get("args") or {}).keys()) if isinstance(tc.get("args"), dict) else [],
                )
        return {
            "pending_ai_message": response,
            "tool_calls_used": state.get("tool_calls_used", [])
            + [{"name": tc.get("name"), "args": tc.get("args")} for tc in toolCalls if isinstance(tc, dict)],
        }
    text = getattr(response, "content", "") or str(response)
    return {"messages": [response], "final_answer": text, "pending_ai_message": None}


async def reasoningMcpNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    llm: Any = None,
    tools: Optional[list] = None,
    settings: Optional[Settings] = None,
    store: Any = None,
) -> dict[str, Any]:
    """
    MCP-only tool phase (no web_search in tool list).
    """
    return await reasoningPhaseNode(state, config, llm=llm, tools=tools, settings=settings, phase="mcp", store=store)


async def reasoningWebNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    llm: Any = None,
    tools: Optional[list] = None,
    settings: Optional[Settings] = None,
    store: Any = None,
) -> dict[str, Any]:
    """
    Web search phase only (typically web_search).
    """
    return await reasoningPhaseNode(state, config, llm=llm, tools=tools, settings=settings, phase="web", store=store)


async def reasoningLegacyNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    llm: Any = None,
    tools: Optional[list] = None,
    settings: Optional[Settings] = None,
    store: Any = None,
) -> dict[str, Any]:
    """
    Single-phase reasoning: MCP + web_search together (when sourced_pipeline is disabled).
    """
    return await reasoningPhaseNode(state, config, llm=llm, tools=tools, settings=settings, phase="legacy", store=store)


async def toolNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Execute MCP tool calls from pending_ai_message; append AIMessage + ToolMessages atomically."""
    settings = settings or getSettings()
    messages = state.get("messages") or []
    pending = state.get("pending_ai_message")
    ai_msg: AIMessage | None = pending if isinstance(pending, AIMessage) else None
    append_full_turn = True

    if ai_msg is None and messages:
        last = messages[-1]
        if _is_ai_message(last) and getattr(last, "tool_calls", None) and isinstance(last, AIMessage):
            ai_msg = last
            append_full_turn = False
            logger.warning("toolNode: legacy path; appending ToolMessages only (assistant already in messages)")

    if ai_msg is None:
        logger.error("toolNode: no AIMessage with tool_calls")
        return {"pending_ai_message": None}

    toolCalls = getattr(ai_msg, "tool_calls", None) or []
    if not toolCalls:
        logger.error("toolNode: AIMessage has no tool_calls")
        return {"pending_ai_message": None}

    names_exec = [str(tc.get("name") or "") for tc in toolCalls if isinstance(tc, dict)]
    last_tool_phase = "web" if any(n == "web_search" for n in names_exec) else "mcp"

    results = list(state.get("tool_results") or [])
    tool_msgs: list[ToolMessage] = []
    web_sources = list(state.get("web_sources") or [])

    for tc in toolCalls:
        name = tc.get("name") or ""
        args = tc.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        tool_call_id = tc.get("id") or tc.get("tool_call_id") or ""
        logger.info("toolNode: executing name=%s tool_call_id=%s", name, tool_call_id)
        try:
            out = await invokeTool(name, args, settings)
            if isinstance(out, dict) and "web_sources" in out:
                entry: dict[str, Any] = {"name": name, "result": out}
                web_sources.extend(out.get("web_sources") or [])
                content = str(out.get("content", json.dumps(out, ensure_ascii=False)))
            else:
                entry = {"name": name, "result": out}
                if isinstance(out, (dict, list)):
                    content = json.dumps(out, ensure_ascii=False)
                else:
                    content = str(out)
            results.append(entry)
            logger.info("toolNode: success tool_call_id=%s", tool_call_id)
        except Exception as e:
            results.append({"name": name, "result": str(e)})
            content = str(e)
            logger.warning("toolNode: error tool_call_id=%s err=%s", tool_call_id, e)
        tool_msgs.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    expected_ids = {str(tc.get("id") or tc.get("tool_call_id") or "") for tc in toolCalls if isinstance(tc, dict)}
    expected_ids.discard("")
    got_ids = {str(getattr(tm, "tool_call_id", "") or "") for tm in tool_msgs}
    got_ids.discard("")
    if expected_ids != got_ids:
        logger.error("toolNode: tool_call_id mismatch expected=%s got=%s", expected_ids, got_ids)

    msg_update: list[BaseMessage] = [ai_msg, *tool_msgs] if append_full_turn else tool_msgs
    out_state: dict[str, Any] = {
        "tool_results": results,
        "messages": msg_update,
        "pending_ai_message": None,
        "last_tool_phase": last_tool_phase,
    }
    if web_sources:
        out_state["web_sources"] = web_sources
    return out_state


async def summarizeNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    llm: Any = None,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """
    Summarize conversation when over threshold: extend existing summary with messages,
    then remove all but the last stm_keep_last_n messages via RemoveMessage.
    """
    settings = settings or getSettings()
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)

    raw_messages = list(state.get("messages") or [])
    existing_summary = state.get("summary") or ""
    keep_last_n = max(0, getattr(settings, "stm_keep_last_n", STM_KEEP_LAST_N))

    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above. "
            "Include a short chronological bullet list of user questions (verbatim phrases) in the summary."
        )
    else:
        prompt = (
            "Summarize the conversation above. "
            "Include a short chronological bullet list of user questions (verbatim phrases)."
        )

    messages_for_summary = raw_messages + [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages_for_summary)
    summary_text = getattr(response, "content", "") or str(response)

    messages_to_delete = raw_messages[:-keep_last_n] if keep_last_n > 0 else raw_messages
    remove_list = [RemoveMessage(id=m.id) for m in messages_to_delete]

    return {"summary": summary_text, "messages": remove_list}


async def answerNode(
    state: dict[str, Any],
    config: RunnableConfig,
    *,
    store: Any = None,
    llm: Any = None,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Generate final answer; system prompt includes LTM user_details when user_id present."""
    settings = settings or getSettings()
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)

    ltm = _ltm_text_for_prompt(store, config) if store is not None else ""
    userDetailsContent = ltm if ltm else "(empty)"

    systemPrompt = LTM_SYSTEM_PROMPT_TEMPLATE.format(user_details_content=userDetailsContent)
    systemPrompt += "\n\nProvide a clear, accurate final answer. If you already have a final_answer in the conversation, you may refine it briefly."
    systemPrompt += (
        "\n\nChat memory: If the user asks what they asked earlier, about previous messages, or a recap, "
        "use only the Human/assistant messages and the conversation summary above. "
        "Do not invent topics (e.g. languages, countries, people) that do not appear there. "
        "Do not say a recap is 'accurate' or 'comprehensive' unless you checked it against those messages. "
        "If the summary lists topics you cannot find in the messages, say so and list only what you can verify."
    )
    sourced_pipeline = state.get("sourced_pipeline", True)
    if sourced_pipeline:
        systemPrompt += (
            "\n\nSourced mode: consolidate only information from final_answer, retrieved context, "
            "tool results, the conversation above, and the user long-term memory block in this system prompt "
            "(user details). That memory is allowed for identity and preferences the user told you—e.g. their name. "
            "Do not invent external facts; prior user/assistant turns in this thread count as allowed context for meta-questions about the chat."
        )

    messages = list(state.get("messages") or [])
    summary = (state.get("summary") or "").strip()
    if summary:
        messages = [SystemMessage(content=f"Conversation summary:\n{summary}")] + messages
    finalAnswer = (state.get("final_answer") or "").strip()
    if finalAnswer:
        response = await llm.ainvoke(
            [SystemMessage(content=systemPrompt)]
            + messages
            + [
                HumanMessage(
                    content=(
                        "Produce the final answer. Draft to use as a starting point (may be wrong for chat-history "
                        "recaps—follow Chat memory rules above):\n"
                        f"{finalAnswer}"
                    )
                )
            ],
        )
    else:
        response = await llm.ainvoke([SystemMessage(content=systemPrompt)] + messages)

    text = getattr(response, "content", "") or str(response)
    return {"messages": [response], "final_answer": text}
