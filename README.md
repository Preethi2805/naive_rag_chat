# **Document-Based Question-Answering System Using Naive RAG**

## **Overview**
This project implements a **Document-Based Question-Answering System** using a **Naive Retrieval-Augmented Generation (RAG)** approach. The system retrieves relevant document chunks from a local database and uses **Groq's LLaMA model** to generate context-aware answers to user queries. 

The system works with text data (news articles in this case) and performs the following tasks:
1. Loads news articles stored as `.txt` files from a specified directory.
2. Splits the content of the articles into smaller chunks to allow for efficient retrieval.
3. Generates embeddings for each chunk using a **SentenceTransformer** model to store and query in **ChromaDB**.
4. Uses **Groq’s API** to generate answers by querying relevant chunks retrieved from the database based on the user’s question.

---

## **Features**

- **Document Loading**: Loads `.txt` files from a specified directory to be processed.
- **Text Chunking**: Splits documents into smaller chunks to ensure that each chunk is of manageable size, helping improve retrieval accuracy.
- **Embedding Generation**: Embeds text chunks into vector representations using **SentenceTransformer** to efficiently store them in the database.
- **Database Storage**: Uses **ChromaDB** for persistent storage and retrieval of embeddings.
- **Question Answering**: Queries the database for relevant chunks based on a user-provided question and uses **Groq’s LLaMA model** to generate contextually relevant answers.

---

## **Explanation of Key Components**

1. **Embedding Generation**:
   - The `SentenceTransformer` model (`sentence-transformers/all-MiniLM-L6-v2`) converts text chunks into numerical vector representations (embeddings). These embeddings capture semantic meaning, enabling accurate document retrieval.

2. **ChromaDB**:
   - ChromaDB is used as a persistent vector database for storing embeddings. It enables efficient querying to find the most relevant document chunks based on a user’s query.

3. **Groq’s LLaMA Model**:
   - **Groq** is used to generate responses from the LLaMA model. The relevant chunks retrieved from ChromaDB are fed into Groq’s model to generate contextually aware answers.

4. **Text Chunking**:
   - Text is split into smaller, overlapping chunks for efficient retrieval and to prevent the model from being overwhelmed by large documents.

5. **Persistent Storage**:
   - The system uses Chroma’s persistent storage to ensure that embeddings are not lost across runs. This allows for future queries to be answered without needing to reprocess documents.

---

## **Project Workflow**

1. **Document Loading**:
   - All `.txt` files from the `news_articles` directory are loaded into memory.

2. **Text Chunking**:
   - Each document is split into smaller chunks for better retrieval performance.

3. **Embedding Generation**:
   - Each chunk is converted into an embedding using the SentenceTransformer model.

4. **Document Insertion**:
   - The embeddings are stored in **ChromaDB** for efficient retrieval during query time.

5. **Querying**:
   - When the user inputs a question, the system retrieves the most relevant chunks using embeddings.

6. **Response Generation**:
   - The retrieved chunks are then passed to **Groq’s LLaMA model** to generate an answer based on the context of the question.

---

## **Dependencies**

- **ChromaDB**: Vector database for storing embeddings.
- **Sentence-Transformers**: For generating text embeddings.
- **Groq**: For interfacing with the LLaMA model to generate responses.
- **python-dotenv**: For securely storing and loading the API key.
- **os**: To interact with the operating system for file handling.

---

## **Future Enhancements**

- **Multi-language support**: Extend the system to support documents and questions in multiple languages.
- **Real-time news updates**: Integrate with an API to fetch real-time news articles and continuously update the document collection.
- **Advanced user interface**: Implement a web-based front end for a more user-friendly interface.
