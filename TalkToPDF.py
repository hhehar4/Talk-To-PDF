# Built by referencing https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/Ask%20A%20Book%20Questions.ipynb

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import pinecone
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


def document_splitter(file_path):
    loader = PyPDFLoader(file_path=file_path)

    data = loader.load()
    print("Document loaded successfully")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    print("Document split successfully")

    return texts


def create_embedding(text):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN,
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    if INDEX_NAME not in pinecone.list_indexes():
        # The all-mpnet-base-v2 model uses cosine as the metric with 768 dimensions
        pinecone.create_index(name=INDEX_NAME, metric="cosine", dimension=768)
        print("Pinecone index created")

    docsearch = Pinecone.from_documents(text, embeddings, index_name=INDEX_NAME)
    print("Embeddings added to index")

    return docsearch


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


def main():
    file_path = "rockets-guide-20-how-rockets-work.pdf"
    # Document loader and splitter
    # Only needs to be run when saving the embeddings into Pinecone for the first time
    texts = document_splitter(file_path)

    # Create embeddings for the loaded document to save into Pinecone
    # Only needs to be run when saving the embeddings into Pinecone for the first time
    docsearch = create_embedding(texts)

    # Ask the LLM to answer the following question based on the vectors saved in Pinecone
    question = "Please list the 3 laws of motion mentioned in the document"
    print(answer_question(question))


if __name__ == "__main__":
    main()
