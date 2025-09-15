from flask import Flask, request, jsonify, render_template
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

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

    # Retrieve top chunks with scores
    results = db.similarity_search_with_score(question, k=5)

    print("Top chunks:")
    for doc, score in results:
        print(doc.page_content[:200], score)

    # Construct context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    # Call LLM
    response_text = model.invoke(prompt)

    # Prepare sources and chunks for front-end
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    chunks_texts = [doc.page_content for doc, _score in results]

    results = db.similarity_search_with_score(question, k=20)


    return jsonify({
        "question": question,
        "answer": response_text,
        "sources": sources,
        "chunks": chunks_texts
    })

if __name__ == "__main__":
    app.run(debug=True)
