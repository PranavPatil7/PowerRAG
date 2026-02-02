# PowerRAG – LLM Driven Knowledge and Analytic Platform                                                     
PowerRAG powerful and user-friendly Streamlit web application designed to facilitate interactions with Large Language Models (LLMs) and manage knowledge bases using Retrieval Augmented Generation (RAG). It allows users to chat directly with local LLMs, query custom RAG memories, and interact with PDF documents.

✨ Features
This application offers four distinct modes of interaction, providing flexibility for various AI-powered tasks:

## Talk to LLM Directly:

Engage in real-time conversations with your selected local Ollama LLM.

Responses stream word-by-word for a dynamic chat experience.

## Talk based on RAG Memory:

Chat with the LLM, with its answers augmented by a persistent knowledge base.

Automatically loads RAG memory (FAISS index) saved from previous sessions.

Displays source documents from which the answer was retrieved, enhancing transparency.

## Talk with PDF (Temporary Memory):

Upload a single PDF document for a temporary Q&A session.

The application processes the PDF, creates embeddings, and builds a vector store in memory for the current session.

Ideal for quick questions about a specific document without saving it permanently.

## Create RAG Memory (Persistent Storage):

Build or update your persistent RAG knowledge base by uploading multiple PDF and/or TXT files.

Documents are processed, embeddings are generated, and the FAISS vector store is saved locally (faiss_index.bin).

Includes an option to delete the persistent RAG memory file.

## Libaries Used
- **Streamlit**: For building the interactive web application GUI.

- **LangChain**: A framework for developing applications powered by language models.

- **langchain-community.llms.Ollama**: For integrating with local Ollama LLMs.

- **langchain-community.embeddings.OllamaEmbeddings**: For generating embeddings using Ollama models.

- **langchain-community.vectorstores.FAISS**: For efficient similarity search and vector database management.

- **langchain.chains.RetrievalQA, langchain.chains.LLMChain**: For orchestrating LLM interactions and RAG.

- **langchain_core.prompts.PromptTemplate**: For structuring prompts for LLMs.

- **Ollama**: A platform for running open-source LLMs locally.

- **PyPDF**: For loading and parsing PDF documents.

- **Requests**: For making HTTP requests to the Ollama server.

 - **Python os & tempfile**: For file system operations and temporary file management.

## Setup Instructions
Follow these steps to get Aby's Assistant up and running on your local machine.

<u>**Prerequisites**</u>
- Python 3.8+: Ensure Python is installed on your system.

- Ollama:Download and install Ollama from ollama.com.


- Start the Ollama server in your terminal:
    ```bash
        ollama serve    # This will start the ollama server in your local system
    ```

- Pull the necessary LLM and Embedding models. For example:
    ```bash
    ollama pull llama2              # A general purpose LLM you try with small model like qwen 0.6b because it do not need much memory.
    ollama pull nomic-embed-text    # A dedicated embedding model

    ```
(You can choose other models as per your preference, but ensure they are pulled.)

<u>**Installation**</u>

- Clone the Repository (if applicable):
    - git clone <your-repo-url>
    - cd <your-repo-name>
    - (If you've been working directly with files, ensure all project files are in one directory.)

<u>**Create a Virtual Environment (Recommended):**</u>
```bash
    python -m venv .venv
```
<u>**Activate the Virtual Environment:**</u>

- Windows:
    ```bash
    .venv\Scripts\activate
    ```
- macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```
<u>**Install Dependencies:**</u>

pip install -r requriments.txt 

# How to Run the Application
- Ensure your virtual environment is activated.

- Navigate to the directory containing your app.py  file.

- Run the Streamlit application:
   ```bash
        streamlit run app.py
    ```

Your browser should automatically open to the Streamlit application.

## Usage Guide
- **Global Settings (Sidebar):**

    -  *Ollama Server URL:* Confirm or update your Ollama server's address.

    -   *Select LLM (Question Answering) Model:* Choose the model you want to use for generating responses (e.g., llama2).

    -   *Select Embedding Model (for PDF/RAG processing):* Choose a dedicated embedding model (e.g., nomic-embed-text). This is crucial for RAG functionality.

    -   *Temperature:* Adjust the creativity of the LLM's responses.

    -   *Choose Interaction Mode (Sidebar Radio Buttons):*

**Talk to LLM:** Type your questions directly into the chat input. The LLM will respond based on its general knowledge.

**Talk based on RAG Memory:**

If you've previously built RAG memory, it will attempt to load automatically.

Ask questions, and the LLM will try to answer using information from your persistent knowledge base.

Source documents will be displayed if found.

**Talk with PDF:**

Upload a PDF document. The app will process it and build a temporary vector store.

Once processed, you can ask questions related to the content of that specific PDF.

This memory is not saved after the session ends or the PDF is cleared.

**Create RAG Memory:**

Upload multiple PDF and/or TXT files.

Click "Build/Update RAG Memory" to process these documents and save them as your persistent RAG knowledge base (faiss_index.bin).

You can also "Delete Persistent RAG Memory File" to clear all saved RAG data.
```bash
Project Structure 
.
├── app.py         
├── llm_chat/         
├    └── direct_llm_chat.pdf    
├── pdf_chat/         
├    └── pdf_qa_session.pdf      
├── rag_chat/         
├    └── rag_chat.pdf      
├── rag_memory/         
├    └── create_rag_memory.pdf           
├── app.py          
├── license.txt      
├── README.md                    
├── requirements.txt        
├── faiss_index.bin      # This will Create When you create your VECTOR Database     
└── rag_documents/       # When You will start Updloading Document For Vector Database    
    └── temp_uploaded_file.pdf

```

### Future Enhancements (Ideas)
**External LLM Integration:** Re-introduce API key inputs for services like Google Gemini or OpenAI.

**Advanced RAG Options:**
- More sophisticated retrieval techniques (e.g., HyDE, RAG-Fusion).
- Support for more document types (e.g., .docx, .csv).
- RAG memory management UI (viewing loaded documents, adding/removing specific docs).

**Chat History Persistence:** Save chat histories across sessions.

**User Authentication:** Implement user login for personalized RAG memories.

**Deployment:** Instructions for deploying to cloud platforms (e.g., Hugging Face Spaces, Render, AWS).

## 📄 License
This project is open-source and available under the MIT License.
