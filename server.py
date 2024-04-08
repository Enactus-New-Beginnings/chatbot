from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request, jsonify
from llama_index.core import (
    VectorStoreIndex,
    Prompt,
    Document,
    ServiceContext,
    set_global_service_context,
)
from flask_cors import CORS, cross_origin
from llama_index.llms.openai import OpenAI
from pdfminer.high_level import extract_text

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)


@app.route("/", methods=["GET"])
def hello():
    print("get")
    return "Hello, World!"


@app.route("/query", methods=["POST"])
def query_model():
    data = request.get_json()
    resume_text = extract_text("get the path_to_your_resume.pdf from firebase")
    resume_doc = Document(text=resume_text)
    index = VectorStoreIndex.from_documents([resume_doc], show_progress=True)

    TEMPLATE_STR = (
        "You are a friendly job search assistant with a speciality in resume building experience.\n"
        "Given my current resume, your purpose is to answer any questions I have about my future career paths, or about the job searching process. If something unrelated to these is asked, say you cannot answer.\n"
        "If I ask you to generate content for a specific job based on my resume, perform a fuzzy search on the lists for information regarding that job.\n"
        "If I ask you how I should improve my resume, ask me what field of career I would like to pursue first, then answer the question.\n"
        "If I ask you how I should gain more experience and increase my chance of employment, ask me what field of career I would like to pursue first, then answer the question.\n"
        #add a few more constrains that only allow the bot to answer resume related or career based questions
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )

    QA_TEMPLATE = Prompt(TEMPLATE_STR)

    query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)

    response = query_engine.query(data["query"])
    response_str = (
        "Seems like you have no resume on file. Navigate to Profile and upload a new resume to start the screening process."
        if str(response) == "None"
        else str(response).replace("[Your Name]", data["name"])
    )
    return jsonify({"response": response_str})


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=3000, url_scheme="https")
