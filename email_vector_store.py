import os
import email
from email import policy
from email.parser import BytesParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from bs4 import BeautifulSoup

# 임베딩 모델과 텍스트 분할기 초기화
embedding_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')  # 다른 모델을 선택할 수 있음
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 전역 변수 초기화
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

                # 이메일 내용 추출
                subject = msg['subject'] if msg['subject'] else ''
                body = msg.get_body(preferencelist=('plain', 'html'))
                if body:
                    try:
                        content = body.get_content()
                    except Exception as e:
                        print(f"본문 추출 중 오류 발생: {e}")

                # 본문이 없을 때 fallback
                if not content:
                    for part in msg.walk():
                        try:
                            if part.get_content_type() == "text/plain":
                                content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            elif part.get_content_type() == "text/html":
                                html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                # HTML에서 순수 텍스트 추출
                                soup = BeautifulSoup(html_content, 'html.parser')
                                content += soup.get_text(separator='\n')  # 줄바꿈으로 구분된 순수 텍스트
                        except Exception as e:
                            print(f"MIME 파트 처리 중 오류 발생: {e}")

                # 여전히 본문이 없으면 전체 페이로드 시도
                if not content:
                    try:
                        content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except Exception as e:
                        print(f"페이로드 추출 중 오류 발생: {e}")

                # 본문이 제대로 추출되었는지 확인
                if content:
                    full_text = f"Subject: {subject}\n\n{content}"
                    # 텍스트를 청크로 분할
                    chunks = text_splitter.split_text(full_text)
                    # 각 청크를 임베딩
                    for chunk in chunks:
                        embedding = embedding_model.encode(chunk)
                        embeddings.append(embedding)
                        documents.append(chunk)

    # 임베딩을 NumPy 배열로 변환
    embeddings_array = np.array(embeddings).astype('float32')

    # FAISS 인덱스 생성
    if embeddings_array.size > 0:
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
    else:
        print("임베딩이 생성되지 않았습니다. .eml 파일에 내용이 있는지 확인하세요.")

    # 인덱스와 문서 저장
    faiss.write_index(index, 'email_index.faiss')
    with open('documents.npy', 'wb') as f:
        np.save(f, np.array(documents))

def load_index_and_documents():
    global index, documents
    if os.path.exists('email_index.faiss') and os.path.exists('documents.npy'):
        index = faiss.read_index('email_index.faiss')
        documents = np.load('documents.npy', allow_pickle=True)
    else:
        print("인덱스와 문서를 찾을 수 없습니다. 생성하려면 process_eml_files()를 실행하세요.")

# 스크립트로 실행될 때 .eml 파일 처리
if __name__ == '__main__':
    process_eml_files()
else:
    # 모듈로 가져올 때 인덱스와 문서를 로드
    load_index_and_documents()
