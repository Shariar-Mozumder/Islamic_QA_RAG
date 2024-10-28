# Install necessary packages
# pip install langchain faiss-cpu sentence-transformers pypdf transformers

import uuid
import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.messages import trim_messages

# 1. Session/User Management
def generate_user_id():
    return str(uuid.uuid4())

user_memory_store = {}

def get_user_memory(user_id):
    if user_id not in user_memory_store:
        user_memory_store[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return user_memory_store[user_id]

# 2. PDF Document Processing - Load Multiple PDFs
def load_pdfs(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory_path, filename))
            documents.extend(loader.load())
    return documents

# 3. Split Documents into Chunks
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# 4. Build FAISS Vector Store
def create_faiss_index(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# 5. Retrieve Relevant Context from FAISS
def get_relevant_context(query, vector_store, k=3):
    results = vector_store.similarity_search(query, k=k)  # Retrieve top k most relevant documents
    context = "\n".join([doc.page_content for doc in results])
    return context

# 6. Text Generation Model Initialization
def init_text_generation_model():
    generator = pipeline("text-generation", model="distilgpt2")
    llm = HuggingFacePipeline(pipeline=generator)
    return llm

# 7. Create LLM Chain with Memory
def create_llm_chain(llm):
    prompt_template = """
    Given the following context and chat history:
    {context}

    User query: {query}

    Answer the question based on the context.
    """
    prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain


# 8. Main Chatbot Function
def agentic_chatbot(user_id, query, vector_store, llm_chain, max_context_tokens=1024, max_new_tokens=100):
    # Step 1: Retrieve user-specific memory
    user_memory = get_user_memory(user_id)

    # Step 2: Retrieve the chat memory and trim if it exceeds max_context_tokens
    trimmed_memory = trim_messages(user_memory.chat_memory.messages, max_context_tokens - len(query.split()))

    # Join trimmed messages to create memory context
    memory_context = "\n\n".join([msg.content for msg in trimmed_memory])

    # Step 3: Retrieve relevant documents from FAISS or vector store
    context = get_relevant_context(query, vector_store)
    
    # Combine both contexts (past conversation memory + new retrieved documents)
    full_context = memory_context + "\n\n" + context

    # Step 4: Generate a response using the LLM chain
    try:
        response = llm_chain.run(
            context=full_context,  # Using the combined context (trimmed memory + new docs)
            query=query,
            max_new_tokens=max_new_tokens,  # Specify tokens for generation
            max_length=max_context_tokens + max_new_tokens  # Adjust max length
        )
    except ValueError as e:
        print(f"Error while generating response: {e}")
        response = "I'm sorry, I encountered an issue processing your request."

    # Step 5: Update user memory with the latest conversation
    user_memory.save_context({"query": query}, {"response": response})

    # Return the generated response
    return response





# 9. Full Chatbot Workflow
if __name__ == "__main__":
    # Load PDFs from a directory (replace 'pdf_directory' with your path)
    pdf_directory = "pdfs"  # Replace this with the actual directory containing your PDFs
    documents = load_pdfs(pdf_directory)

    # Split documents into chunks
    split_docs = split_documents(documents)

    # Create FAISS vector store from document chunks
    vector_store = create_faiss_index(split_docs)

    # Initialize text generation model
    llm = init_text_generation_model()

    # Create the LLM chain with memory
    llm_chain = create_llm_chain(llm)

    # Example chatbot session
    print("Welcome to the Agentic RAG Chatbot!")
    user_id = generate_user_id()  # Start a new session for the user

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        # Get chatbot response
        response = agentic_chatbot(user_id, user_query, vector_store, llm_chain)
        print(f"Chatbot: {response}")
