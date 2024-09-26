import numpy as np
import requests
from email_vector_store import index, embedding_model, documents

def retrieve(query, index, embedding_model, documents, top_k=5):
    query_embedding = embedding_model.encode(query).astype('float32')
    distances, indices = index.search(np.array([query_embedding]), top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    url = f'http://192.168.0.66:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    data = { 
        "model": "llama3.1:70b-instruct-q4_0",
        "prompt": prompt,
        "stream": False
    }

    llm_res = requests.post(url, headers=headers, json=data)
    llm_res = llm_res.json()
    answer = llm_res.get("response", "None").strip()
    print(f"[모델 출력] : {answer}")
    return answer

def chatbot(query):
    retrieved_docs = retrieve(query, index, embedding_model, documents)
    answer = generate_answer(query, retrieved_docs)
    return answer