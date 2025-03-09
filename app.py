# To interact with the operating system
import os 

# To load envirnment variables from the .env file
from dotenv import load_dotenv 

# Database for storing and querying embeddings generated from the text data
import chromadb

# Imports Groq's API to interact with models
from groq import Groq

from sentence_transformers import SentenceTransformer

# To create embedding functions to convert text to numerical vector representations.
from chromadb.utils import embedding_functions

# Load envirnment variables from .env file
load_dotenv()

# Setting up the key from the .env file - Used to authenticate requests to OpenAI's API
ai_key = os.getenv("API_KEY")

local_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to generate embeddings
def embed_text(text):
    return local_embedding_model.encode(text).tolist()  # Convert to list for ChromaDB

# Iniitialising a persistent client for ChromaDB
chroma_client = chromadb.PersistentClient(path = "chroma_persistent_storage")
# path = "chroma_persistent_storage" sets the director where the embeddings will be stored persistently(data isnt lost when program restarts)

collection_name = "document_qa_collection" # Name of the collection in the ChromaDB - embeddings are stored here

# Checking if the collection with the name already exists, if not, it is created.
collection = chroma_client.get_or_create_collection(
    name = collection_name) # embedding_functions = openai_ef specifies the function used to convert the text to embeddings

# Initialising an OpenAI client to make API calls to OpenAI
client = Groq(api_key = ai_key) # Used to send requests to OpenAI's models

# Testing out the model
""" resp  =  client.chat.completions.create(
    model = "llama3-8b-8192",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the human life expectancy in the United States?",
        },
    ],
)

print(resp.choices[0].message.content) """

# Loading documents from a directory
def load_documents(directory_path):
    print("==== Loading documents from directory ====")
    documents = []

    # Looping through each document in the directory_path
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            # If it ends with .txt, opens the file, reads its content and saves it in the list
            with open(
                os.path.join(directory_path, filename), "r", encoding = "utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split the text into chucks 
def split_text(text, chuck_size = 1000, chunk_overlap = 100): # ensures overlapping parts to preserve context between the chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chuck_size
        chunks.append(text[start:end])
        # Moving start forward with an overlap for continuity
        start = end - chunk_overlap
    return chunks

# Load documents from the dictionary
directory_path = "./news_articles"
documents = load_documents(directory_path)

print(f"Loaded {len(documents)} documents.")

# Splitting each document into chucks 
chucked_documents = []

# Looping through each document 
print("==== Splitting docs into chunks ====")
for docs in documents:
    chucks = split_text(docs["text"]) # Extracting the text contents and breaking them into smaller chucks 
    
    # For each chuck, a dictionary is created with id(document id_chunk number) and text
    for i, chuck in enumerate(chucks):
        chucked_documents.append({"id": f"{docs['id']}_chuck{i+1}", "text": chuck})

print(f"Split documents into {len(chucked_documents)} chucks")

# Generating embeddings 
def get_embeddings(text): 
    # Generating embeddings locally 
    response = local_embedding_model.encode(text).tolist()
    return response

# Looping through every chunk 
print("==== Generating Embeddings ====")
for doc in chucked_documents:
    # Generating a embedding for each chunk
    doc["embedding"] = get_embeddings(doc["text"]) # Another key in doc apart from "id" and "text"

# Upsert(Insert/Update) documents with embedding into Chroma
print("==== Inserting chucks into db ====")
for doc in chucked_documents:
    collection.upsert(
        ids = [doc["id"]], documents = [doc["text"]], embeddings = [doc["embedding"]]
    )

#print(doc["embedding"])

# Function to query the stored documents - Pass the question and number of results we want to receive
def query_documents(question, n_results = 5):

    query_embedding = get_embeddings(question)
    
    # Getting the results - By querying the closest matches to the query text
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    # ChromaDB compares the embedding of the query with the embeddings of documents stored in the collection.
    # n_result limits the number of results returned from the database
    # Returns a list of lists (list of chucks of relevant documents)

    # Flattening the results['documents'] - By iterating over all the sublists and pulling out individual chunks, this ensures that you get all the document chunks that match your query, preserving their relevance to the question you asked.
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chucks ====")
    return relevant_chunks

# Function to generate the response from LLM
def generate_response(question, relevant_chunks):
    # Joining the relevant chucks into a single string seperated by 2 newlines - for structuring
    context = "\n\n".join(relevant_chunks) 

    # prompt = system instruction + question + context
    prompt = ( 
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question 
    )
    
    # Generating the response by calling the LLM API
    response = client.chat.completions.create(
        model = "llama3-8b-8192",
        messages = [
            # System message - instruction
            {
                "role" : "system",
                "content" : prompt,
            },
            # User message - Actual question
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer

question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)