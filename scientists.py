from langchain_community.retrievers import WikipediaRetriever
from langgraph.graph import StateGraph
from groq import Groq
from typing import TypedDict
from prompts import S0_START_PROMPT, S1_PROMPT, S2_PROMPT, S3_PROMPT
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ScientistState(TypedDict):
    topic: str
    s1_response: str
    s2_response: str
    s3_response: str
    final_abstract: str
    additional_notes: str


class Scientist:
    def __init__(self, name, agent, prompt):
        self.name = name
        self.agent = agent
        self.prompt = prompt
        self.response = None
        self.retriever = WikipediaRetriever()

    def query_tool(self, topic):
        """Ensure the topic is passed correctly as 'input'."""
        print(f"{self.name} is querying the agent for the topic '{topic}'...")

        # Correct usage of invoke with 'input' parameter
        retrieved_docs = self.retriever.invoke(input=topic)  # Pass 'topic' as 'input'
        formatted_docs = "\n".join([doc.page_content for doc in retrieved_docs])

        # Construct the prompt with the retrieved data
        full_prompt = f"{self.prompt}\n\nRelevant Data:\n{formatted_docs}"
        print(f"Querying {self.name} with prompt:\n{full_prompt}")

        # Generate the response using the agent
        completion = self.agent.chat.completions.create(
            model="llama3-8b-8192",  # Replace with the appropriate model
            messages=[{"role": "system", "content": full_prompt}],
            temperature=1,
            max_tokens=1500,
        )
        self.response = completion.choices[0].message.content.strip()
        return self.response


def generate_embeddings(s1_response, s2_response, s3_response):
    """Generate embeddings for the responses from S1, S2, and S3."""
    embeddings_generator = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Wrap the responses from S1, S2, and S3 in Document objects
    document_s1 = Document(page_content=s1_response)
    document_s2 = Document(page_content=s2_response)
    document_s3 = Document(page_content=s3_response)

    # Extract the page_content (text) from the Document objects
    text_s1 = document_s1.page_content
    text_s2 = document_s2.page_content
    text_s3 = document_s3.page_content

    # Split the documents into chunks if needed (optional based on size of the text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    
    # Combine the text responses into a list for splitting
    documents = [text_s1, text_s2, text_s3]
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings for each chunk of the documents
    embeddings = embeddings_generator.embed_documents(chunks)
    
    return embeddings

































# from groq import Groq
# from langchain_community.retrievers import WikipediaRetriever
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env file
# load_dotenv()

# # Access environment variables
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# # In scientists.py
# class Scientist:
#     def __init__(self, name, agent, prompt):
#         self.name = name
#         self.agent = agent
#         self.prompt = prompt
#         self.response = None
#         self.retriever = WikipediaRetriever()

#     def query_tool(self, topic):
#         """Ensure the topic is passed correctly as 'input'."""
#         print(f"{self.name} is querying the agent for the topic '{topic}'...")

#         # Correct usage of invoke with 'input' parameter
#         retrieved_docs = self.retriever.invoke(input=topic)  # Pass 'topic' as 'input'
#         formatted_docs = "\n".join([doc.page_content for doc in retrieved_docs])

#         # Construct the prompt with the retrieved data
#         full_prompt = f"{self.prompt}\n\nRelevant Data:\n{formatted_docs}"
#         print(f"Querying {self.name} with prompt:\n{full_prompt}")

#         # Generate the response using the agent
#         completion = self.agent.chat.completions.create(
#             model="llama3-8b-8192",  # Replace with the appropriate model
#             messages=[{"role": "system", "content": full_prompt}],
#             temperature=1,
#             max_tokens=1500,
#         )
#         self.response = completion.choices[0].message.content.strip()
#         return self.response
