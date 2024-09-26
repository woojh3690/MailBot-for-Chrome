import os
import email
from email import policy
from email.parser import BytesParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 텍스트 분할기와 임베딩 모델 초기화
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 다른 모델을 선택할 수 있음

# 임베딩과 문서를 저장할 리스트 준비
embeddings = []
documents = []

# 메시지 폴더 경로
messages_folder = 'messages'

for filename in os.listdir(messages_folder):
    if filename.endswith('.eml'):
        filepath = os.path.join(messages_folder, filename)
        with open(filepath, 'rb') as file:
            msg = BytesParser(policy=policy.default).parse(file)
            # 이메일 내용 추출
            subject = msg['subject'] if msg['subject'] else ''
            body = msg.get_body(preferencelist=('plain', 'html'))
            if body:
                content = body.get_content()
                full_text = f"Subject: {subject}\n\n{content}"
                # 텍스트를 청크로 분할
                chunks = text_splitter.split_text(full_text)
                # 각 청크 임베딩
                for chunk in chunks:
                    embedding = embedding_model.encode(chunk)
                    embeddings.append(embedding)
                    documents.append(chunk)

# 임베딩을 NumPy 배열로 변환
embeddings = np.array(embeddings).astype('float32')

# FAISS 인덱스 구축
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 인덱스와 문서 저장
faiss.write_index(index, 'email_index.faiss')
with open('documents.npy', 'wb') as f:
    np.save(f, np.array(documents))
