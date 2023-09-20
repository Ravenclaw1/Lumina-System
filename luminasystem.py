#!/usr/bin/env python3

# Import required libraries and modules
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time

# Load environment variables from a .env file
if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# Extract necessary configurations from environment variables
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

# Import chroma settings from a constants module
from constants import CHROMA_SETTINGS

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize embeddings and the Chroma database client
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Configure callbacks based on the command-line arguments
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Initialize the Language Learning Model (LLM) based on the specified model type
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    # Initialize the QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    # Get the user's name for a more personalized interaction
    username = input("\nHello! I am Kiran. Before we begin, may I know your name? ")

    # Main interaction loop
    while True:
        query = input(f"\n{username}, enter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Time the process of getting an answer
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Display the result and sources (if any)
        print(f"\n\n> {username}:")
        print(query)
        print(f"\n> Kiran: (took {round(end - start, 2)} s.):")
        print(answer)

        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='The Lumina System: a digital infrastructure created by Nova. It hosts the digital entity Kiran.'
                                                'Kiran learns to provide assistance over sources.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

# If the script is executed directly, run the main function
if __name__ == "__main__":
    main()
