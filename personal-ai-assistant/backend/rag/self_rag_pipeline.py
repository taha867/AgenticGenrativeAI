"""
Self-RAG pipeline (architecture from self_rag_step7).
LangGraph: decide_retrieval -> retrieve -> is_relevant -> generate_from_context -> is_sup -> revise_answer / is_use -> rewrite_question.
"""
from typing import Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from backend.core.config import Settings, getSettings
from backend.core.constants import MAX_RETRIES
from backend.rag.retriever import formatDocs

# State (TypedDict) - match step7
class SelfRagState(TypedDict, total=False):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: list[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str


# Pydantic decision models
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(..., description="True if external documents are needed.")


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(..., description="True if doc contains info that can answer the question.")


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: list[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="Short reason in 1 line.")


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(..., description="Rewritten query for vector retrieval.")


def buildSelfRagGraph(
    llm: BaseChatModel,
    retriever: BaseRetriever,
    settings: Optional[Settings] = None,
    *,
    disallow_direct_generation: bool = False,
):
    """Build and compile the Self-RAG StateGraph.

    When disallow_direct_generation is True, never routes to generate_direct (no parametric-only answers).
    """
    from langgraph.graph import END, StateGraph

    s = settings or getSettings()
    maxRewriteTries = s.self_rag_max_refine_rounds

    # Prompts (full text from self_rag_step7.ipynb for strict IsSUP, quote-only revise, topic-level relevance)
    decidePrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON with key: should_retrieve (boolean).\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts from company documents.\n"
            "- should_retrieve=False for general explanations/definitions.\n"
            "- If unsure, choose True.",
        ),
        ("human", "Question: {question}"),
    ])
    directPrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer using only your general knowledge.\n"
            "If it requires specific company info, say:\n"
            "'I don't know based on my general knowledge.'",
        ),
        ("human", "{question}"),
    ])
    relevancePrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are judging document relevance at a TOPIC level.\n"
            "Return JSON matching the schema.\n\n"
            "A document is relevant if it discusses the same entity or topic area as the question.\n"
            "It does NOT need to contain the exact answer.\n\n"
            "Examples:\n"
            "- HR policies are relevant to questions about notice period, probation, termination, benefits.\n"
            "- Pricing documents are relevant to questions about refunds, trials, billing terms.\n"
            "- Company profile is relevant to questions about leadership, culture, size, or strategy.\n\n"
            "Do NOT decide whether the document fully answers the question.\n"
            "That will be checked later by IsSUP.\n"
            "When unsure, return is_relevant=true.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ])
    ragPrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a business rag chatbot.\n\n"
            "You will receive a CONTEXT block from internal company documents.\n"
            "Task:\n"
            "Answer the question based on the context. "
            "Do not mention that you are getting a context in your answer.",
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ])
    issupPrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}"),
    ])
    revisePrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.",
        ),
        ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nCONTEXT:\n{context}"),
    ])
    isusePrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ])
    rewritePrompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}",
        ),
        ("human", "QUESTION:\n{question}\n\nPrevious retrieval query:\n{retrieval_query}\n\nAnswer (if any):\n{answer}"),
    ])

    decideLlm = llm.with_structured_output(RetrieveDecision)
    relevanceLlm = llm.with_structured_output(RelevanceDecision)
    issupLlm = llm.with_structured_output(IsSUPDecision)
    isuseLlm = llm.with_structured_output(IsUSEDecision)
    rewriteLlm = llm.with_structured_output(RewriteDecision)

    def decideRetrieval(state: SelfRagState) -> dict:
        decision = decideLlm.invoke(decidePrompt.format_messages(question=state["question"]))
        return {"need_retrieval": decision.should_retrieve}

    def routeAfterDecide(state: SelfRagState) -> Literal["generate_direct", "retrieve"]:
        if disallow_direct_generation:
            return "retrieve"
        return "retrieve" if state.get("need_retrieval") else "generate_direct"

    def generateDirect(state: SelfRagState) -> dict:
        out = llm.invoke(directPrompt.format_messages(question=state["question"]))
        return {"answer": out.content}

    def retrieve(state: SelfRagState) -> dict:
        q = state.get("retrieval_query") or state["question"]
        docs = retriever.invoke(q)
        return {"docs": docs}

    def isRelevant(state: SelfRagState) -> dict:
        relevant = []
        for doc in state.get("docs", []):
            decision = relevanceLlm.invoke(relevancePrompt.format_messages(question=state["question"], document=doc.page_content))
            if decision.is_relevant:
                relevant.append(doc)
        return {"relevant_docs": relevant}

    def routeAfterRelevance(state: SelfRagState) -> Literal["generate_from_context", "no_answer_found"]:
        return "generate_from_context" if (state.get("relevant_docs") and len(state["relevant_docs"]) > 0) else "no_answer_found"

    def generateFromContext(state: SelfRagState) -> dict:
        rel = state.get("relevant_docs", [])
        context = formatDocs(rel)
        if not context:
            return {"answer": "No answer found.", "context": ""}
        out = llm.invoke(ragPrompt.format_messages(question=state["question"], context=context))
        return {"answer": out.content, "context": context}

    def noAnswerFound(state: SelfRagState) -> dict:
        return {"answer": "No answer found.", "context": ""}

    def isSup(state: SelfRagState) -> dict:
        decision = issupLlm.invoke(issupPrompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        ))
        return {"issup": decision.issup, "evidence": decision.evidence}

    def routeAfterIssup(state: SelfRagState) -> Literal["is_use", "revise_answer"]:
        if state.get("issup") == "fully_supported":
            return "is_use"
        if state.get("retries", 0) >= MAX_RETRIES:
            return "is_use"
        return "revise_answer"

    def reviseAnswer(state: SelfRagState) -> dict:
        out = llm.invoke(revisePrompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        ))
        return {"answer": out.content, "retries": state.get("retries", 0) + 1}

    def isUse(state: SelfRagState) -> dict:
        decision = isuseLlm.invoke(isusePrompt.format_messages(question=state["question"], answer=state.get("answer", "")))
        return {"isuse": decision.isuse, "use_reason": decision.reason}

    def routeAfterIsuse(state: SelfRagState) -> Literal["__end__", "rewrite_question", "no_answer_found"]:
        if state.get("isuse") == "useful":
            return "__end__"
        if state.get("rewrite_tries", 0) >= maxRewriteTries:
            return "no_answer_found"
        return "rewrite_question"

    def rewriteQuestion(state: SelfRagState) -> dict:
        decision = rewriteLlm.invoke(rewritePrompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        ))
        return {
            "retrieval_query": decision.retrieval_query,
            "rewrite_tries": state.get("rewrite_tries", 0) + 1,
            "docs": [],
            "relevant_docs": [],
            "context": "",
        }

    builder = StateGraph(SelfRagState)
    builder.add_node("decide_retrieval", decideRetrieval)
    builder.add_node("generate_direct", generateDirect)
    builder.add_node("retrieve", retrieve)
    builder.add_node("is_relevant", isRelevant)
    builder.add_node("generate_from_context", generateFromContext)
    builder.add_node("no_answer_found", noAnswerFound)
    builder.add_node("is_sup", isSup)
    builder.add_node("revise_answer", reviseAnswer)
    builder.add_node("is_use", isUse)
    builder.add_node("rewrite_question", rewriteQuestion)

    builder.add_conditional_edges(
        "decide_retrieval",
        routeAfterDecide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )
    builder.add_edge("generate_direct", END)
    builder.add_edge("retrieve", "is_relevant")
    builder.add_conditional_edges(
        "is_relevant",
        routeAfterRelevance,
        {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"},
    )
    builder.add_edge("generate_from_context", "is_sup")
    builder.add_edge("no_answer_found", END)
    builder.add_conditional_edges(
        "is_sup",
        routeAfterIssup,
        {"is_use": "is_use", "revise_answer": "revise_answer"},
    )
    builder.add_edge("revise_answer", "is_sup")

    def routeAfterIsuseFixed(state: SelfRagState) -> Literal["__end__", "rewrite_question", "no_answer_found"]:
        if state.get("isuse") == "useful":
            return "__end__"
        if state.get("rewrite_tries", 0) >= maxRewriteTries:
            return "no_answer_found"
        return "rewrite_question"

    builder.add_conditional_edges(
        "is_use",
        routeAfterIsuseFixed,
        {"__end__": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"},
    )
    builder.add_edge("rewrite_question", "retrieve")
    builder.set_entry_point("decide_retrieval")
    return builder.compile()


async def runSelfRag(
    query: str,
    retriever: BaseRetriever,
    llm: BaseChatModel,
    settings: Optional[Settings] = None,
    *,
    disallow_direct_generation: bool = False,
) -> tuple[str, list[dict], int, int, bool]:
    """
    Run Self-RAG pipeline asynchronously.
    Returns (answer, sources, rewrite_tries, retries, retrieval_from_documents).
    retrieval_from_documents is True iff final state has non-empty relevant_docs.
    """
    graph = buildSelfRagGraph(
        llm, retriever, settings, disallow_direct_generation=disallow_direct_generation
    )
    initialState: SelfRagState = {
        "question": query,
        "retrieval_query": query,
        "rewrite_tries": 0,
        "retries": 0,
        "answer": "",
        "context": "",
        "relevant_docs": [],
    }
    config = RunnableConfig(recursion_limit=80)
    finalState = await graph.ainvoke(initialState, config=config)
    answer = finalState.get("answer") or "No answer found."
    relevantDocs = finalState.get("relevant_docs") or []
    sources = [{"content": d.page_content[:500], "metadata": d.metadata} for d in relevantDocs]
    rewriteTries = finalState.get("rewrite_tries", 0)
    retries = finalState.get("retries", 0)
    retrieval_from_documents = len(relevantDocs) > 0
    return answer, sources, rewriteTries, retries, retrieval_from_documents
