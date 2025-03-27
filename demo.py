import json
import os
import pathlib
from uuid import uuid4
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import chromadb

# Load environment variables
load_dotenv()

# Define paths
filepath = pathlib.Path().resolve()
DATASET = str(filepath) + "/data_sources"

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store
persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("summary")

# Initialize vector store client
vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="summary",
    embedding_function=embeddings,
)


# Load JSON data
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# Convert data to Document format
def document_converter(data):
    dataset = []
    for item in data:
        dataset.append(
            Document(page_content=item["text"])
        )  # Assuming 'text' is the key in your dataset
    return dataset


# Chatbot response function
def chatbot_response(question):
    # Retrieve similar documents
    document = vector_store_from_client.similarity_search(question, k=2)

    # Define prompt template
    prompt_template = """
    Sen bir politikacısın ve olaylara karşı karar alacaksın.
    {document}
    Olay: {question}
    """
    template = PromptTemplate(
        input_variables=["document", "question"],
        template=prompt_template,
    )

    # Initialize LLM (Ollama)
    llm = ChatOllama(model="llama3.2", temperature=0)

    # Initialize memory
    memory = ConversationBufferMemory(input_key="question", memory_key="chat_history")

    # Initialize LLMChain
    chain = LLMChain(llm=llm, prompt=template, memory=memory, verbose=True)

    # Generate response
    response = chain.run(document=document, question=question)
    return response


# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Main function
def main():
    try:
        # Load dataset
        dataset = load_dataset(
            "text", data_files={"train": f"{DATASET}/datasets/trnews-64.val.raw"}
        )
        print("Dataset loaded successfully.")

        # Print dataset structure for debugging
        print("Dataset structure:", dataset)

        # Tokenize the dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        print("Dataset tokenized successfully.")

        # Convert dataset to Document format
        converted_dataset = document_converter(tokenized_dataset["train"])
        print("Dataset converted to Document format.")

        # Add documents to vector store
        uuids = [str(uuid4()) for _ in range(len(converted_dataset))]
        batch_size = 100
        for i in range(0, len(converted_dataset), batch_size):
            batch = converted_dataset[i : i + batch_size]
            vector_store_from_client.add_documents(
                documents=batch, ids=uuids[i : i + batch_size]
            )
        print("Documents added to vector store.")

        # Chatbot loop
        print("\nYou can close the chatbot by typing 'exit'.")
        history = ChatMessageHistory()
        while True:
            query = input("Type your question: ")
            if query.lower() == "exit":
                print("Chatbot is closing.")
                break

            # Get chatbot response
            response = chatbot_response(query)
            history.add_user_message(query)
            print(response)
            history.add_ai_message(response)

    except Exception as e:
        print(f"Error: {e}")


# Run the main function
if __name__ == "__main__":
    main()
