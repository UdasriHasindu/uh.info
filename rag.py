import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Load environment variables from .env file
load_dotenv()

# Verify API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found!")
    print("Please create a .env file with: GOOGLE_API_KEY=your-api-key-here")
    exit(1)


# 1. Load document
loader = TextLoader("info.txt")
document = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(document)


# 3. Create embedding (`Google gemini `)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# 4. Vector store (`chromadb`)
CHROMA_DIR = "./chroma_langchain_db"

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# Check if documents already exist in the collection
existing_docs = vector_store.get()
if not existing_docs['ids']:
    # Collection is empty, add documents
    print("Adding documents to vector store...")
    vector_store.add_documents(documents=all_splits)
    print(f"Added {len(all_splits)} documents")
else:
    print(f"Found existing collection with {len(existing_docs['ids'])} documents")

# Retrieves top 3 chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# 5. RAG chain


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# Create a prompt template
template = """You are an assistant answering questions about Udasri Hasindu's professional profile.
Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know.
Keep the answer concise and relevant.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    
    if query.lower() == "exit":
        break
    
    response = rag_chain.invoke(query)
    print("\nAnswer:\n", response)




