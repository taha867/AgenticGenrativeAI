import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyB2hY_0DVDCFwRgGxwuLMFYXa8Hz51IsYY"

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Step 1a - Indexing (Document Ingestion)

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "Gfr50f6ZBvo"
print("Step 1a - Indexing (Document Ingestion)")
print("\n")

try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=["en"])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

print("\n\n")

# Step 1b - Indexing (Text Splitting)

print("Step 1b - Indexing (Text Splitting)")
print("\n")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

print(len(chunks))
print("\n\n")

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
# Free tier: 100 embed requests/min. Process in batches with delay to avoid 429.
import time

print("Step 1c - Indexing Embedding Generation")
print("\n")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
print(embeddings)
print("\n\n")

print("Step 1d - Indexing Storing in Vector Store")
print("\n")

BATCH_SIZE = 50
vector_store = None
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    if vector_store is None:
        vector_store = FAISS.from_documents(batch, embeddings)
    else:
        vector_store.add_documents(batch)
    if i + BATCH_SIZE < len(chunks):
        print(f"Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks, waiting 35s for rate limit...")
        time.sleep(35)

print(vector_store.index_to_docstore_id)
print(vector_store.get_by_ids(["81e2869d-6542-451c-ab06-03b39e4532ab"]))

print("\n\n")

# Step 2 - Retrieval

print("Step 2 - Retrieval")
print("\n")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

print(retriever)
print("\n")
print(retriever.invoke('What is deepmind'))
print("\n")
## Step 3 - Augmentation

print("Step 3 - Augmentation")
print("\n")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

print(retrieved_docs)
print("\n")
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(context_text)
print("\n")
final_prompt = prompt.invoke({"context": context_text, "question": question})
print(final_prompt)
print("\n")
# Step 4 - Generation

print("Step 4 - Generation")
print("\n")

answer = llm.invoke(final_prompt)
print(answer.content)
print("\n\n")

# Building a Chain

print("Building a Chain")
print("\n")

answer = llm.invoke(final_prompt)
print(answer.content)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

main_chain.invoke('Can you summarize the video')
print(main_chain.content)
print("\n\n")