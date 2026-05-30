import os
from pathlib import Path
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Step 1: Setup
load_dotenv()
print("API Key Loaded.")

docs_dir = Path("groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"


# Step 2: Load the Documents
print("\n--- Loading Documents ---")
documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"Total documents loaded: {len(documents)}")
for doc in documents:
    print(f"Loaded: {doc.metadata.get('file_name', 'Unknown')}")


# Step 3: Build Index and query engine
print("\n--- Building Index ---")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")


# Step 4: Query assistant
print("\n--- Querying Assistant ---")
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print(f"A: {response.response}")
    
    top_node = response.source_nodes[0]
    filename = top_node.node.metadata.get('file_name', 'Unknown')
    print(f"Top Source: [{filename}] | Score: {top_node.score:.4f}")
    print(f"Chunk Preview: {top_node.node.text[:200]}...")

"""
Step 4 Reflection:
The assistant sounded highly confident and incredibly accurate. It didn't hallucinate or use overly robotic language. Because it uses semantic search, it easily navigated the "dairy-free" question (likely matching it to almond or oat milk in the menu document), which was a major limitation when we built standard keyword RAG in the warmup.
"""

# Step 5: Find a Failure
print("\n--- Testing Failure Mode ---")
bad_query = "What is the manager's name and how much do you charge for a large vanilla latte?"
bad_resp = query_engine.query(bad_query)

print(f"\nQ: {bad_query}")
print(f"A: {bad_resp.response}\n")

for i, node in enumerate(bad_resp.source_nodes):
    filename = node.node.metadata.get('file_name', 'Unknown')
    print(f"Source {i+1}: [{filename}] | Score: {node.score:.4f}")
    print(f"Preview: {node.node.text[:200]}...\n")

"""
Step 5 Comment:
1. What I asked: I asked for a specific manager's name and an exact price for a vanilla latte. I expected this to be hard because those hyper-specific details are almost never stored in high-level company overviews or menu summaries.
2. What went wrong: The retrieval pulled the menu document and hiring documents (because they matched terms like 'latte' and staff roles), but the exact information was missing from the text. 
3. Model's Tone: The model successfully maintained a polite, helpful tone but explicitly stated it could not provide those details based on the context. It did not hallucinate a fake price or name. This proves we can generally trust it *not* to lie when it lacks data.
4. What to change: I would implement a "fallback protocol" in the system prompt. If the model cannot find the answer, it should automatically provide the store's general phone number or contact email so the user isn't left at a dead end.
"""

# Step 6: Reflection
"""
--- Final Reflection ---

1. Lines of Code Comparison:
Building semantic RAG manually involves dozens of lines of code to handle tokenization, chunking, embedding generation APIs, cosine similarity math, and prompt injection. In LlamaIndex, the entire core pipeline took exactly three lines of code: `SimpleDirectoryReader`, `VectorStoreIndex.from_documents()`, and `.as_query_engine()`. This demonstrates the massive value of frameworks: they abstract away the plumbing, letting you focus on the application logic and user experience rather than managing vector math.

2. Alternative Use Case:
A highly valuable use case would be for a technical IT support desk at a large corporation. The company could point LlamaIndex at their massive repository of internal IT troubleshooting PDFs and past ticketing logs. When an employee asks "How do I connect to the VPN on a Mac?", the RAG assistant instantly pulls the exact internal instructions and delivers the answer, preventing a human IT worker from having to answer the same repetitive question.

3. Unpreventable Failure Mode:
One failure mode RAG cannot fully prevent is answering with outdated information if the underlying documents aren't updated. If Groundwork Coffee changed their weekend hours to close at 4 PM, but the `hours.txt` document wasn't replaced in the database, the RAG system will confidently, and "accurately" (based on its context), lie to the user. RAG is only as good as the hygiene of the documents it is searching.
"""