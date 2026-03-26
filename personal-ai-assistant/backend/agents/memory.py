"""
Long-term memory (LTM) extraction: Pydantic models and prompt for the remember node.
Store only is_new=True items in PostgresStore; namespace ("user", user_id, "details").
"""
from typing import List

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """Atomic user memory; is_new=True means not duplicate of existing."""

    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")


class MemoryDecision(BaseModel):
    """LLM structured output for the remember node."""

    should_write: bool = Field(description="Whether to write any memory")
    memories: List[MemoryItem] = Field(default_factory=list, description="Atomic user memories to store")


MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- If the user states their name, preferred name, or how to address them (e.g. "I'm ...", "call me ...", "my name is ..."),
  you MUST store it as a memory (e.g. "User's name is ...") with is_new=true unless it is already in CURRENT USER DETAILS.
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

# Notebook-style LTM system prompt for answer node: personalization, address by name, suggest 3 questions.
LTM_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant with memory capabilities.
If user-specific memory is available, use it to personalize 
your responses based on what you know about the user.

Your goal is to provide relevant, friendly, and tailored 
assistance that reflects the user's preferences, context, and past interactions.

If the user's name or relevant personal context is available, always personalize your responses by:
    – Always Address the user by name (e.g., "Sure, Nitish...") when appropriate
    – Referencing known projects, tools, or preferences (e.g., "your MCP server python based project")
    – Adjusting the tone to feel friendly, natural, and directly aimed at the user

Avoid generic phrasing when personalization is possible.

Use personalization especially in:
    – Greetings and transitions
    – Help or guidance tailored to tools and frameworks the user uses
    – Follow-up messages that continue from past context

Always ensure that personalization is based only on known user details and not assumed.

If the user asks for their own name or identity and it appears in the user memory above, answer with it.
Do not refuse with "no access to personal information" when the user explicitly shared that information and it is listed in memory above.

In the end suggest 3 relevant further questions based on the current response and user profile.

The user's memory (which may be empty) is provided as: {user_details_content}
"""
