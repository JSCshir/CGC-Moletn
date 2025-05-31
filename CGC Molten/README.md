My RAG Project
==============

A Retrieval-Augmented Generation (RAG) project that processes PDF files, computes embeddings,
and uses a chatbot interface to answer questions based on the content of those PDFs.

-------------------------------------------------------------------------------
Prerequisites
-------------------------------------------------------------------------------
1. Python 3.9+  
   Verify installation by running:
       python3 --version

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------
1. Clone the repository or download the project files into a directory, e.g., /home/jacobhardy/my_rag_project.

2. (Optional but Recommended) Create and activate a virtual environment:
       cd /home/jacobhardy/my_rag_project
       python3 -m venv venv
       source venv/bin/activate

3. Install Required Packages:  
   The project requires the following packages (listed in requirements.txt):
       - fastapi
       - uvicorn
       - sentence-transformers
       - spacy
       - pymupdf
       - pandas
       - tqdm
       - torch
       - langchain_ollama
   Install them by running:
       pip install -r requirements.txt

4. Download the spaCy English model:
       python3 -m spacy download en_core_web_sm

-------------------------------------------------------------------------------
Project Structure
-------------------------------------------------------------------------------
Your project directory should look like this:

my_rag_project/
├── rag/                    
│   ├── __init__.py
│   ├── pdf_processing.py    
│   ├── embeddings.py        
│   ├── retrieval.py         
│   └── chat_chain.py        
├── documents/          # Place your PDF files here (ensure they have a .pdf extension)
├── rag.py              # CLI entry-point
├── server.py           # API server entry-point
└── static/
    └── index.html      # Front-end file

-------------------------------------------------------------------------------
Running the Project
-------------------------------------------------------------------------------

CLI Mode:
---------
1. Ensure your PDF files are in the "documents" folder.
2. Navigate to the project root:
       cd /home/jacobhardy/my_rag_project
3. Run the CLI script:
       python3 rag.py

Server Mode:
------------
1. Ensure your static files (e.g., index.html) are in the "static" folder.
2. Navigate to the project root:
       cd /home/jacobhardy/my_rag_project
3. Run the server using Uvicorn:
       uvicorn server:app --reload
   The server will start at: http://127.0.0.1:8000

-------------------------------------------------------------------------------
Testing the Application
-------------------------------------------------------------------------------
- Front-End:
  Open a web browser and navigate to:  
       http://127.0.0.1:8000  
  You should see your index.html front-end.

- API Documentation:
  Open a web browser and navigate to:  
       http://127.0.0.1:8000/docs  
  This displays the interactive API documentation provided by FastAPI.

-------------------------------------------------------------------------------
Troubleshooting
-------------------------------------------------------------------------------
- PDF Location:  
  Ensure your PDF files are placed in the "documents" folder.

- Correct Python Version:  
  Verify you are using Python 3.9+.

- Dependencies:  
  If you encounter dependency issues, double-check your virtual environment and re-run:
       pip install -r requirements.txt

- Static Files:  
  If the front-end does not load, confirm that index.html is located inside the "static" folder
  and that your server is correctly configured to serve static files.

-------------------------------------------------------------------------------
Happy Coding!





*** 
If you have everything installed, it's as easy as running rag.py once, typing exit when prompted, typing uvicorn server:app --reload into bash, and opening index.html with a live server!
***