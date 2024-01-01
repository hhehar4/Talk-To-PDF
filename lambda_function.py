import json
import pinecone
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
INDEX_NAME = os.getenv("INDEX_NAME")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV,
)


def answer_question(question):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN,
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    if INDEX_NAME in pinecone.list_indexes():
        docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)

        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 64},
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
        )

        return qa.run(question)

    else:
        return "Pinecone index not found"


def lambda_handler(event, context):
    body = json.loads(event["body"])
    question = body["userprompt"]
    response = answer_question(question)

    return {"statusCode": 200, "body": json.dumps(response)}
