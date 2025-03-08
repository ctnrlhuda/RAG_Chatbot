from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface  import HuggingFaceEmbeddings

# Step 1: Initialize Ollama Model
llm = OllamaLLM(model="llama3")  # Ensure the Llama2 model is available in Ollama

# Step 3: Load and Split Documents
docs_dir = r"C:\temp\TMDb_Certifications2.txt"  # Directory containing your knowledge base
loader = TextLoader(docs_dir)
documents = loader.load()

# Use HuggingFace embeddings (free alternative to OpenAIEmbeddings)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build a FAISS vectorstore
vectorstore = FAISS.from_documents(documents, embeddings)

exit_conditions = (":q", "quit", "exit")

# print("Welcome! Please ask your questions below. Type ':q', 'quit', or 'exit' to stop.")

while True:
    query = input("Please ask a question: ")
    if query in exit_conditions:
        print("Goodbye!")
        break
    else:
        try:
            related_docs = vectorstore.similarity_search(query)

            response = llm.invoke(f"Context: {related_docs[0].page_content}\n\nQuestion: {query}")
            print("Answer:", response)
        except Exception as e:
            print(f"An error occurred: {e}")

#Query Llama3 with a similarity search
# query = "What is the document about?"
# related_docs = vectorstore.similarity_search(query)

# # Use Llama3 to generate a response
# response = llm.invoke(f"Context: {related_docs[0].page_content}\n\nQuestion: {query}")
# print("Answer:", response)