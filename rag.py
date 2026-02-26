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
import shutil

CHROMA_DIR = "./chroma_langchain_db"

# Always rebuild DB on startup
if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)

vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name="example_collection",
    persist_directory=CHROMA_DIR,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Retrieves top 3 chunks
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

# 5. RAG chain


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
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

# Interactive mode - only runs when script is executed directly
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RAG System Ready! Ask questions about Udasri Hasindu.")
    print("Type 'exit' to quit.")
    print("="*60)
    
    while True:
        query = input("\n🔍 Ask a question: ")
        
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        if not query.strip():
            continue
        
        response = rag_chain.invoke(query)
        print(f"\n💡 Answer:\n{response}")


