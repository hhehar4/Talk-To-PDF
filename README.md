# Talk-To-PDF

A simple 'Talk to PDF' function which lets you select a PDF document to upload and have an LLM answer your questions using the document as context.

It utilizes the sentence-transformers/all-mpnet-base-v2 model to generate the embeddings for the document which are then saved in a Pinecone index.

The model was chosen as it provided the best quality embeddings out of the SBERT models: https://sbert.net/docs/pretrained_models.html

The model can be found here: https://huggingface.co/sentence-transformers/all-mpnet-base-v2

The LLM used to generate the question-answer chains is google/flan-t5-xxl which can be found here: https://huggingface.co/google/flan-t5-xxl

The TalkToPDF.py file contains the code used to load and split the PDF document, create and save the embeddings for the document to Pinecone, and answer questions using the saved document as context.

The lambda_function.py file contains the code which is being run by the AWS Lambda function to receive a POST request containing the question which then responds with the answer after querying the LLM using the same answer question function used in the TalkToPDF.py file.
