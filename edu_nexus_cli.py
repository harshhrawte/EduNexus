import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─── Load environment variables ─────────────────────────────────
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# ─── Validate GROQ key ──────────────────────────────────────────
if not GROQ_KEY.startswith("gsk_") or len(GROQ_KEY) != 56:
    raise ValueError("Invalid GROQ_API_KEY – check your .env file.")

# ─── Configure Chroma DB ────────────────────────────────────────
persist_dir = "chroma_db"
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Initialize PersistentClient for Chroma DB
client = PersistentClient(path=persist_dir)
collection = client.get_collection(name="langchain")

# Check if collection has data
if collection.count() == 0:
    print("⚠ Warning: Chroma DB 'langchain' is empty. Using model knowledge only.")

# ─── Configure Groq Llama 3 8B model ────────────────────────────
llm = ChatGroq(
    groq_api_key=GROQ_KEY,
    model_name="Llama3-8b-8192",
    temperature=0.5,
    max_tokens=1024,
    max_retries=3,
    request_timeout=30
)

# ─── Define RAG prompt ──────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_template(
    """You are a friendly school teacher helping students from grades 5–10.
Answer the question in a clear, simple, and concise way (2–3 sentences unless more detail is requested) that a young student can understand.
Use the provided textbook excerpts only if they directly relate to the question.
If the excerpts are irrelevant or missing, rely entirely on your own knowledge to provide an accurate answer.
Do NOT mention missing or unhelpful excerpts—just give the best answer possible.
If you are unsure, say “I’m not sure, let me check,” but avoid this unless absolutely necessary.

Textbook Excerpts:
{context}

Question: {query}
Answer:"""
)

# ─── Interactive CLI ────────────────────────────────────────────
def main():
    print("\n🎓 Welcome to EduNexus – Your Student Q&A Companion!")
    print("------------------------------------------------------")
    
    while True:
        q = input("\n❓ Ask a question (or type 'exit'): ").strip()
        
        # Handle exit command
        if not q or q.lower() in ("exit", "quit"):
            print("👋 Goodbye! Keep learning and stay curious.")
            break
        
        # Ensure question ends with a question mark
        if not q.endswith("?"):
            q += "?"

        print("\n🧠 Thinking...\n")
        
        try:
            # Retrieve documents from Chroma DB
            embedding_vector = embeddings.embed_query(q)
            results = collection.query(
                query_embeddings=[embedding_vector],
                n_results=3
            )
            
            # Process retrieved documents with relevance check
            context = ""
            if results["documents"] and results["documents"][0]:
                # Basic relevance check: ensure documents contain question-related keywords
                keywords = q.lower().split()
                relevant_docs = [
                    doc for doc in results["documents"][0]
                    if any(keyword in doc.lower() for keyword in keywords)
                ]
                if relevant_docs:
                    print(f"📚 Retrieved Documents: {len(relevant_docs)} relevant documents found")
                    for i, doc in enumerate(relevant_docs, 1):
                        print(f"  Doc {i}: {doc[:100]}...")
                        context += f"- {doc}\n"
                else:
                    print("⚠ No relevant documents found. Using model knowledge.")
                    context = ""
            else:
                print("⚠ No relevant documents found. Using model knowledge.")
                context = ""

            # Debug input
            print(f"📝 Passing input to model: query = {q}")
            
            # Invoke Llama 3 8B with RAG prompt
            response = llm.invoke(rag_prompt.format(context=context, query=q))
            
            # Debug raw response
            print(f"🔍 Raw Answer: {response.content}")
            
            # Display answer
            print(f"📘 Answer:\n{response.content.strip()}\n")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("🛠 Tips:")
            print("- Check your GROQ_API_KEY and network connection.")
            print("- Ensure Chroma DB is accessible.")

if __name__ == "__main__":
    main()