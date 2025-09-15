from flask import Flask, request, jsonify, render_template
import os
from populate_database import load_documents, split_documents, add_to_chroma, DATA_PATH
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
import shutil

app = Flask(__name__)

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

Question: {question}
"""

# DB and model
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = OllamaLLM(model="mistral")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    results = db.similarity_search_with_score(question, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    chunks_texts = [doc.page_content for doc, _ in results]

    return jsonify({
        "question": question,
        "answer": response_text,
        "sources": sources,
        "chunks": chunks_texts
    })


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf_file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded PDF into the DATA_PATH
    file_path = os.path.join(DATA_PATH, file.filename)
    file.save(file_path)

    # Run your populate_database logic
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    return jsonify({"message": f"PDF '{file.filename}' processed and added to database."})


if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    app.run(debug=True)
