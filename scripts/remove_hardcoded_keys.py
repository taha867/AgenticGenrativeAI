#!/usr/bin/env python3
"""Replace hardcoded OPENAI_API_KEY with load_dotenv() in known files. Run from repo root.
Used when rewriting git history (e.g. during rebase) to remove secrets from a commit."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "langChain/Tools/tool_calling_in_langchain.py",
    "langChain/RAGcomponents/retrivers/langchain_retrievers.py",
    "langChain/RAGcomponents/vectorStore/langchain_chroma.py",
]

for rel in FILES:
    path = ROOT / rel
    if not path.exists():
        continue
    text = path.read_text()
    # Remove line that sets OPENAI_API_KEY to a secret
    text = re.sub(r'\nos\.environ\["OPENAI_API_KEY"\] = "sk-[^"]*"\n', "\n", text)
    # After "import os" add dotenv (if not already there)
    if "load_dotenv" not in text and "import os" in text:
        text = text.replace(
            "import os\n",
            "import os\nfrom dotenv import load_dotenv\nload_dotenv()  # from .env\n",
            1,
        )
    path.write_text(text)
    print("Fixed:", rel)
