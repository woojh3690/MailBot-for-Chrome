# email_vector_store.py

import os
import email
from email import policy
from email.parser import BytesParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize embedding model and text splitter
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose another model
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Initialize global variables
index = None
documents = []
embeddings = []

def process_eml_files(messages_folder='messages'):
    global embeddings, documents, index

    embeddings = []
    documents = []

    for filename in os.listdir(messages_folder):
        if filename.endswith('.eml'):
            filepath = os.path.join(messages_folder, filename)
            with open(filepath, 'rb') as file:
                msg = BytesParser(policy=policy.default).parse(file)
                # Extract email content
                subject = msg['subject'] if msg['subject'] else ''
                body = msg.get_body(preferencelist=('plain', 'html'))
                if body:
                    content = body.get_content()
                    full_text = f"Subject: {subject}\n\n{content}"
                    # Split text into chunks
                    chunks = text_splitter.split_text(full_text)
                    # Embed each chunk
                    for chunk in chunks:
                        embedding = embedding_model.encode(chunk)
                        embeddings.append(embedding)
                        documents.append(chunk)

    # Convert embeddings to NumPy array
    embeddings_array = np.array(embeddings).astype('float32')

    # Build FAISS index
    if embeddings_array.size > 0:
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
    else:
        print("No embeddings were generated. Check if the .eml files contain content.")

    # Save the index and documents
    faiss.write_index(index, 'email_index.faiss')
    with open('documents.npy', 'wb') as f:
        np.save(f, np.array(documents))

def load_index_and_documents():
    global index, documents
    if os.path.exists('email_index.faiss') and os.path.exists('documents.npy'):
        index = faiss.read_index('email_index.faiss')
        documents = np.load('documents.npy', allow_pickle=True)
    else:
        print("Index and documents not found. Please run process_eml_files() to generate them.")

# Load index and documents when imported as a module
load_index_and_documents()

# When executed as a script, process the .eml files
if __name__ == '__main__':
    process_eml_files()
