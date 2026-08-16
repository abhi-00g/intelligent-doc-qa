import google.generativeai as genai
import os

SYSTEM_PROMPT = """You are a helpful assistant. Answer concisely using only 
the supplied context. If the answer is not in the context, say you don't know. 
Cite the source document when possible."""

def answer_with_gemini(question: str, context_chunks):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "Error: GEMINI_API_KEY not set in your .env file."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    context = "\n\n".join([f"- {c[:1200]}" for c in context_chunks])
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
    
    response = model.generate_content(prompt)
    return response.text.strip()