import os
import time
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Flask setup
app = Flask(__name__)
CORS(app)

# Create upload directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
retriever = None
chat_history = []
upload_progress = 0

@app.route("/")
def index():
    return "PDF QA Assistant API is running!"

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "retriever_ready": retriever is not None,
        "progress": upload_progress
    })

@app.route("/uploadprogress", methods=["POST"])
def upload_progress_route():
    global chat_history, upload_progress
    
    print("Upload request received")
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files allowed"}), 400
    
    # Save file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"File saved: {filepath}")
    
    # Reset state
    chat_history = []
    upload_progress = 0
    
    # Start processing in background
    thread = threading.Thread(target=process_pdf, args=(filepath,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Upload successful, processing started"})

@app.route("/progress")
def progress():
    def generate():
        global upload_progress
        last_sent = -1
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_progress = upload_progress
            
            if current_progress != last_sent:
                yield f"data: {current_progress}\n\n"
                last_sent = current_progress
                print(f"Progress sent: {current_progress}%")
            
            if current_progress >= 100 or current_progress < 0:
                break
                
            time.sleep(0.5)
        
        # Send final progress
        yield f"data: {upload_progress}\n\n"
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    return response

@app.route("/ask", methods=["POST"])
def ask():
    global retriever, chat_history
    
    print("Question request received")
    
    if not retriever:
        return jsonify({"error": "No document uploaded"}), 400
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question required"}), 400
    
    question = data['question'].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    
    print(f"Processing question: {question}")
    
    try:
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({"error": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."}), 500
        
        # Create chain
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = ConversationalRetrievalChain.from_llm(llm, retriever)
        
        # Get answer
        result = chain.invoke({
            "question": question, 
            "chat_history": chat_history[-5:]  # Keep last 5 exchanges
        })
        
        answer = result["answer"]
        chat_history.append((question, answer))
        
        print(f"Answer generated: {answer[:100]}...")
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

def process_pdf(filepath):
    global retriever, upload_progress
    
    try:
        print(f"Starting PDF processing: {filepath}")
        upload_progress = 10
        
        # Read PDF
        reader = PdfReader(filepath)
        total_pages = len(reader.pages)
        print(f"PDF has {total_pages} pages")
        
        if total_pages == 0:
            print("Error: PDF has no pages")
            upload_progress = -1
            return
        
        upload_progress = 20
        
        # Extract text
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                # Update progress 20-60%
                upload_progress = 20 + int((i + 1) / total_pages * 40)
                time.sleep(0.1)  # Small delay for progress visualization
            except Exception as e:
                print(f"Error extracting text from page {i+1}: {e}")
                continue
        
        if not text.strip():
            print("Error: No text extracted from PDF")
            upload_progress = -1
            return
        
        print(f"Text extracted: {len(text)} characters")
        upload_progress = 65
        
        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")
        upload_progress = 75
        
        if not chunks:
            print("Error: No chunks created")
            upload_progress = -1
            return
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        upload_progress = 85
        
        # Create vector store
        print("Creating vector store...")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        upload_progress = 95
        
        print("PDF processing completed successfully!")
        upload_progress = 100
        
        # Clean up file
        try:
            os.remove(filepath)
            print(f"Cleaned up: {filepath}")
        except:
            pass
            
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        upload_progress = -1

if __name__ == "__main__":
    print("Starting PDF QA Assistant server...")
    print("Make sure to set OPENAI_API_KEY environment variable!")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not found!")
        print("   Please set it: export OPENAI_API_KEY='your-api-key-here'\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)