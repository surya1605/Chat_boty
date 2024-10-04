import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Simple in-memory vector store
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, doc, embedding):
        self.documents.append(doc)
        self.embeddings.append(embedding)

    def search(self, query_embedding, top_k=1):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

# Initialize components
vectorizer = TfidfVectorizer()
vector_store = SimpleVectorStore()
generator = pipeline('text-generation', model='gpt2')

# Sample documents
documents = [
    "The sky is blue.",
    "Grass is green.",
    "The sun is yellow."
]

# Create embeddings and add to vector store
doc_embeddings = vectorizer.fit_transform(documents).toarray()
for doc, embedding in zip(documents, doc_embeddings):
    vector_store.add_document(doc, embedding)

# RAG function
def rag(query):
    # Convert query to embedding
    query_embedding = vectorizer.transform([query]).toarray()[0]
    
    # Retrieve relevant documents
    relevant_docs = vector_store.search(query_embedding, top_k=2)
    
    # Generate response
    context = " ".join([doc for doc, _ in relevant_docs])
    prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    return response

# Example usage
query = "What color is the sky?"
result = rag(query)
print(f"Query: {query}")
print(f"Response: {result}")