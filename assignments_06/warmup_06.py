import os
import string
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# ==========================================
# --- RAG Concepts ---
# ==========================================

"""
# Concepts Q1
Scenario A: RAG (Retrieval-Augmented Generation). The legal team has hundreds PDFs that change quarterly. RAG allows  model to search these exact documents dynamically without needing to be retrained every time a policy changes.
Scenario B: Fine-tuning. The startup has 3,000 specific examples of a unique brand voice. Fine-tuning actually adjusts the model's internal weights to adopt this specific tone and style across all future outputs.
Scenario C: Prompt Engineering (Context Injection). The analyst only has a two-page report. This small enough to simply paste into the prompt alongside the question. No database or retraining is needed.

# Concepts Q2
Why confidently wrong is worse: A model that says "I am not sure" prompts the user to double-check facts or look elsewhere. A confidently wrong answer (hallucination) tricks the user into believing false information, which can lead to disastrous decisions. 
Example of harm: If an AI medical assistant confidently hallucinates a medication dosage, a nurse might administer a lethal dose, trusting the system.
Tone and trust: Humans are socially conditioned to associate confident tone and fluent delivery with expertise and accuracy. When an LLM sounds authoritative, we naturally drop our guard and trust it more.

# Concepts Q3
Steps in a complete RAG pipeline:
1. "Extract text from source documents" - Read the raw text out of PDFs, Word docs, etc.
2. "Split text into chunks" - Break the large documents into smaller, searchable paragraphs.
3. "Convert text chunks into embeddings" - Turn the chunks into number vectors using an embedding model.
4. "Receive the user's query" - The user asks a question.
5. "Embed the user's query" - Turn the user's question into a vector using the same embedding model.
6. "Retrieve the most relevant chunks" - Find the chunks with vectors mathematically closest to the query vector.
7. "Inject retrieved chunks into the prompt" - Combine the user's question and the retrieved text chunks into one prompt.
8. "Generate a response from the LLM" - Send the combined prompt to the LLM for the final answer.
"""


# ==========================================
# --- Keyword RAG ---
# ==========================================
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)
    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")
    
    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")
            
    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Keyword Q1
print("\n--- Keyword Q1 ---")
query_1 = "What are your hours on the weekend?"
simple_keyword_retrieval(query_1, documents, verbose=True)
"""
Keyword Q1 Comment: The system selected 'hours.txt'. It stripped out the stopwords and punctuation, matching the tokens 'hours' and 'weekend' directly with the exact same words present in hours.txt.
"""

# Keyword Q2
print("\n--- Keyword Q2 ---")
query_2 = "Do you have anything without caffeine?"
simple_keyword_retrieval(query_2, documents, verbose=True)
"""
Keyword Q2 Comment: The system found NO relevant content. Keyword RAG completely failed here because the word "caffeine" does not explicitly exist in 'menu.txt' (even though espresso and lattes implicitly have caffeine, and cold brew is mentioned). Semantic RAG would do much better here because it would understand the conceptual relationship between "without caffeine" and the menu items, rather than looking for exact word overlaps.
"""

# Keyword Q3
"""
Keyword Q3 Prediction: 
Prediction: I predict it will return 'None found' or grab the wrong document.
Reasoning: The user is asking about "rewards" and "sign up". The loyalty.txt document talks about "loyalty program", "earn", and "points", but it does NOT contain the exact words "rewards" or "sign up".
"""
print("\n--- Keyword Q3 ---")
query_3 = "How do I sign up for rewards?"
simple_keyword_retrieval(query_3, documents, verbose=True)
"""
Keyword Q3 Result Reflection: My prediction was correct. No overlapping keywords were found. The vocabulary mismatch between the user's query ("rewards", "sign up") and the document text ("loyalty", "join") caused standard keyword search to fail entirely.
"""

# ==========================================
# --- Semantic RAG Concepts ---
# ==========================================

"""
# Semantic Q1
1. A vector embedding is a mathematical representation of a piece of text (like a word, sentence, or chunk) expressed as a long list of numbers. These numbers map the text into a multi-dimensional space based on its context and meaning.
2. The chunk with a score of 0.85 is much more relevant. Cosine similarity ranges from -1 to 1; a score of 0.85 means the two vectors are pointing in almost the exact same direction, indicating highly similar meaning, whereas 0.30 indicates very little semantic relationship.
3. Semantic search captures the *meaning* (context) rather than exact characters. If a query says "canine" and a document says "dog", their embeddings will end up very close to each other in vector space because they appear in similar contexts during model training, allowing the system to match them despite having zero letters in common.

# Semantic Q2
| Feature                    | Keyword RAG                       | Semantic RAG |
|----------------------------|-----------------------------------|--------------|
| What is compared?          | Exact word overlap                | Vector embeddings (meaning) |
| What is retrieved?         | Full document                     | Text chunks |
| Can it handle synonyms?    | No                                | Yes |
| Storage format             | Plain text dictionary             | Vector store / index |
| Relevance score            | Number of overlapping keywords    | Cosine similarity |
"""


# ==========================================
# --- LlamaIndex ---
# ==========================================


PDF_DIR = "brightleaf_pdfs"

if os.path.exists(PDF_DIR):
    print("\n--- LlamaIndex Q1 ---")
    documents_llama = SimpleDirectoryReader(PDF_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents_llama)
    
    query_engine_3 = index.as_query_engine(similarity_top_k=3)
    
    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?"
    ]
    
    for q in questions:
        print(f"\nQuestion: {q}")
        response = query_engine_3.query(q)
        print(f"Answer: {response.response}\n")
        
        for i, node in enumerate(response.source_nodes):
            print(f"  Node {i+1} Score: {node.score:.4f}")
            print(f"  Node {i+1} Text: {node.node.text[:150]}...\n")
            
    """
    LlamaIndex Q1 Comment:
    1. Do the retrieved chunks look relevant? Yes, the chunks pulled for benefits mention healthcare, PTO, etc. The chunks for security mention data protocols.
    2. Model tone? The model sounds highly confident and specific, listing exactly what is stated in the text. Because it is given the direct context, it doesn't need to hedge.
    3. Unexpected retrievals? Occasionally, a slightly unrelated chunk might get pulled in at position 3 if it shares general corporate terminology, but the LLM is usually smart enough to ignore it and use the top 1 or 2 chunks for the answer.
    """

    print("\n--- LlamaIndex Q2 ---")
    q2 = "What employee benefits does BrightLeaf offer?"
    
    # Top K = 1
    query_engine_1 = index.as_query_engine(similarity_top_k=1)
    res_1 = query_engine_1.query(q2)
    print("\nTop K=1 Answer:", res_1.response)
    print("Top K=1 Score:", res_1.source_nodes[0].score)
    
    # Top K = 5
    query_engine_5 = index.as_query_engine(similarity_top_k=5)
    res_5 = query_engine_5.query(q2)
    print("\nTop K=5 Answer:", res_5.response)
    for i, node in enumerate(res_5.source_nodes):
        print(f"  Top K=5 Node {i+1} Score: {node.score:.4f}")

    """
    LlamaIndex Q2 Comment:
    When k=1, the model might give an incomplete answer if the full list of benefits spans across multiple paragraphs/chunks. 
    When k=5, the response might be identical to k=3, or slightly more detailed. However, more context is NOT always better. Pulling in 5 chunks increases token costs, slows down the response time, and increases the risk of "distracting" the model with irrelevant context that happened to get swept up in the retrieval.
    """

    print("\n--- LlamaIndex Q3 ---")
    q3_hard = "Who is the CEO's favorite band?"
    res_hard = query_engine_3.query(q3_hard)
    print("\nHard Question Answer:", res_hard.response)
    for i, node in enumerate(res_hard.source_nodes):
        print(f"  Node {i+1} Score: {node.score:.4f} | Text: {node.node.text[:100]}...")

    """
    LlamaIndex Q3 Comment:
    Expected: I expected the system to confidently state it doesn't know.
    Happened: The retrieval grabbed 3 totally irrelevant chunks (likely just general company info) because it had to grab *something*, but the LLM looked at those chunks and correctly realized the answer wasn't there, responding with "The provided context does not contain information about..."
    System Change: To improve this, we could add a threshold to the retriever (e.g., only return chunks with a cosine similarity > 0.75). If no chunks meet the threshold, bypass the LLM entirely and immediately tell the user "No relevant documents found."
    """

    print("\n--- LlamaIndex Q4 ---")
    # Setting up the evaluators
    llm = OpenAI(model="gpt-4o-mini")
    faithfulness = FaithfulnessEvaluator(llm=llm)
    relevancy = RelevancyEvaluator(llm=llm)

    # Evaluate Good Query
    print(f"\nEvaluating: {questions[0]}")
    good_resp = query_engine_3.query(questions[0])
    
    f_eval = faithfulness.evaluate_response(response=good_resp)
    r_eval = relevancy.evaluate_response(query=questions[0], response=good_resp)
    print(f"Faithfulness Score: {f_eval.score}")
    print(f"Relevancy Score: {r_eval.score}")

    # Evaluate Bad Query
    print(f"\nEvaluating: {q3_hard}")
    f_eval_bad = faithfulness.evaluate_response(response=res_hard)
    r_eval_bad = relevancy.evaluate_response(query=q3_hard, response=res_hard)
    print(f"Faithfulness Score: {f_eval_bad.score}")
    print(f"Relevancy Score: {r_eval_bad.score}")

    """
    LlamaIndex Q4 Comment:
    1. A faithfulness score of 1.0 means the generated answer is entirely supported by the retrieved context (no hallucinations). A score of 0.0 means the LLM completely ignored the context and hallucinated or made up facts.
    2. Relevancy measures whether the generated answer actually answers the user's original *question*. It differs from faithfulness because an answer can be 100% faithful to a retrieved document, but completely irrelevant to what the user asked.
    3. The scores likely changed. The bad query might have a high faithfulness (if it said "The documents don't state this", which is technically true to the empty context), but a low relevancy because it couldn't provide the answer. 
    4. "LLM-as-a-judge" uses a highly capable LLM (like GPT-4) to grade the output of the RAG pipeline. We use this because natural language answers cannot be evaluated with standard math metrics (like accuracy or F1 scores); you need an AI that understands nuance and context to verify if a generated paragraph accurately reflects a source document.
    """
else:
    print(f"Please update PDF_DIR to point to brightleaf_pdfs to run the LlamaIndex section! Looked at: {PDF_DIR}")