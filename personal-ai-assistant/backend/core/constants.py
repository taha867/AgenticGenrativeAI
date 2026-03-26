"""
Literal constants shared across domains.
Defaults for RAG, memory, and STM summarization; used when not overridden by env.
"""
# Self-RAG
MAX_RETRIES = 10  # IsSUP revise loop cap

# RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 5
SELF_RAG_MAX_REFINE_ROUNDS = 3

# Memory
STM_MAX_TOKENS_DEFAULT = None  # Max tokens for STM trimming; None = unused

# STM summarization (threshold-based trim + summary)
STM_SUMMARIZE_AFTER_MESSAGES = 10  # Run summarization when len(messages) > this; 0/None = disabled
STM_KEEP_LAST_N = 8  # After summarization, keep this many most recent messages verbatim (too low loses chat recall)

# Document upload (API /documents/upload)
UPLOAD_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".markdown"})
