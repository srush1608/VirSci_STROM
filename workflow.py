from langgraph.graph import StateGraph, START, END
from langchain_community.retrievers import WikipediaRetriever
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from scientists import Scientist, generate_embeddings
from groq import Groq
from database import store_query_response
from prompts import S0_START_PROMPT, S1_PROMPT, S2_PROMPT, S3_PROMPT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from scientists import ScientistState
import numpy as np


def start(state):
    return state


def query_agent_s0(state):
    """Leader Agent (S0) queries for the topic and initiates the process."""
    groq_agent = Groq()
    scientist_s0 = Scientist("S0", groq_agent, S0_START_PROMPT)

    # Correctly pass the topic as input to invoke()
    wikipedia_retriever = WikipediaRetriever()
    retrieved_data = wikipedia_retriever.invoke(state["topic"])  # Pass the topic here as input
    formatted_data = "\n".join([doc.page_content for doc in retrieved_data])

    # Construct the prompt for Groq with retrieved Wikipedia data
    full_prompt = f"S0, summarize the following information:\n{formatted_data}"

    # Initialize Groq and query it
    completion = groq_agent.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "system", "content": full_prompt}],
        temperature=1,
        max_tokens=1500,
    )
    s0_response = completion.choices[0].message.content.strip()
    return {'additional_notes': s0_response}


def query_agent_s1(state):
    """Agent S1: Queries the topic and generates insights based on the historical context."""
    groq_agent = Groq()
    scientist_s1 = Scientist("S1", groq_agent, S1_PROMPT)

    # Correctly pass the topic as input to invoke()
    wikipedia_retriever = WikipediaRetriever()
    retrieved_data = wikipedia_retriever.invoke(state["topic"])  # Pass the topic here as input
    formatted_data = "\n".join([doc.page_content for doc in retrieved_data])

    # Construct the prompt for Groq with retrieved Wikipedia data
    full_prompt = f"S1, summarize the following information:\n{formatted_data}"

    # Initialize Groq and query it
    completion = groq_agent.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "system", "content": full_prompt}],
        temperature=1,
        max_tokens=1500,
    )
    response = completion.choices[0].message.content.strip()
    return {'s1_response': response}


def query_agent_s2(state):
    """Agent S2: Queries the topic and provides technical insights."""
    groq_agent = Groq()
    scientist_s2 = Scientist("S2", groq_agent, S2_PROMPT)

    # Correctly pass the topic as input to invoke()
    wikipedia_retriever = WikipediaRetriever()
    retrieved_data = wikipedia_retriever.invoke(state["topic"])  # Pass the topic here as input
    formatted_data = "\n".join([doc.page_content for doc in retrieved_data])

    # Construct the prompt for Groq with retrieved Wikipedia data
    full_prompt = f"S2, summarize the following information:\n{formatted_data}"

    # Initialize Groq and query it
    completion = groq_agent.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "system", "content": full_prompt}],
        temperature=1,
        max_tokens=1500,
    )
    response = completion.choices[0].message.content.strip()
    return {'s2_response': response}


def query_agent_s3(state):
    """Agent S3: Queries the topic and provides ethical or societal insights."""
    groq_agent = Groq()
    scientist_s3 = Scientist("S3", groq_agent, S3_PROMPT)

    # Correctly pass the topic as input to invoke()
    wikipedia_retriever = WikipediaRetriever()
    retrieved_data = wikipedia_retriever.invoke(state["topic"])  # Pass the topic here as input
    formatted_data = "\n".join([doc.page_content for doc in retrieved_data])

    # Construct the prompt for Groq with retrieved Wikipedia data
    full_prompt = f"S3, summarize the following information:\n{formatted_data}"

    # Initialize Groq and query it
    completion = groq_agent.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[{"role": "system", "content": full_prompt}],
        temperature=1,
        max_tokens=1500,
    )
    response = completion.choices[0].message.content.strip()
    return {'s3_response': response}


def interview_state(state, max_rounds=3):
    """
    Conduct multiple rounds of interviews and clarifications between S0 and scientists S1, S2, and S3.
    Ensures new questions are dynamically generated each round based on the latest responses.
    """
    groq_agent = Groq()

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Conversation Round {round_num} ---\n")

        # Step 1: S0 generates new clarifying questions
        interview_prompt_s0 = f"""
        You are S0, the leader scientist, conducting an interview to refine and clarify the topic: '{state['topic']}'.

        Current responses (Round {round_num - 1}):
        - S1 (Historical Context): {state['s1_response']}
        - S2 (Technical Analysis): {state['s2_response']}
        - S3 (Ethical Implications): {state['s3_response']}

        Include all prior notes or clarifications:
        {state.get('additional_notes', '')}

        Generate 2-3 NEW clarifying questions to improve the understanding of the topic.
        Avoid repeating any previous questions. Provide the questions in a numbered list.
        """

        # S0 generates new clarifying questions
        completion_s0 = groq_agent.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": interview_prompt_s0}],
            temperature=0.7,
            max_tokens=300,
        )
        clarifying_questions = completion_s0.choices[0].message.content.strip()
        print(f"S0's Questions (Round {round_num}):\n{clarifying_questions}")

        # Step 2: S1, S2, and S3 refine their responses based on S0's new questions
        updated_s1_response = refine_response(state['s1_response'], clarifying_questions, "S1")
        updated_s2_response = refine_response(state['s2_response'], clarifying_questions, "S2")
        updated_s3_response = refine_response(state['s3_response'], clarifying_questions, "S3")

        # Update the state with refined responses
        state['s1_response'] = updated_s1_response
        state['s2_response'] = updated_s2_response
        state['s3_response'] = updated_s3_response

        # Step 3: Scientists ask new follow-up questions for S0
        scientists_questions = ""
        for scientist, response in zip(
            ["S1", "S2", "S3"], [updated_s1_response, updated_s2_response, updated_s3_response]
        ):
            scientist_prompt = f"""
            You are {scientist}, generating new follow-up questions to ask S0 based on your latest refined response.

            Latest Refined Response (Round {round_num}):
            {response}

            Generate 1-2 NEW questions for S0 to clarify or guide further exploration of the topic.
            Avoid repeating any previous questions. Provide the questions in a numbered list.
            """
            completion = groq_agent.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "system", "content": scientist_prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            questions = completion.choices[0].message.content.strip()
            print(f"{scientist}'s Questions for S0 (Round {round_num}):\n{questions}\n")
            scientists_questions += f"\n{scientist}'s Questions:\n{questions}"

        # Step 4: S0 responds to scientists' follow-up questions
        response_to_scientists_prompt = f"""
        You are S0, the leader scientist. Respond to the following new questions asked by your team members (Round {round_num}):

        {scientists_questions}

        Provide clear and concise answers to help your team refine their understanding.
        """
        completion_s0_response = groq_agent.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": response_to_scientists_prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        s0_additional_notes = completion_s0_response.choices[0].message.content.strip()
        print(f"S0's Responses to Scientists (Round {round_num}):\n{s0_additional_notes}")

        # Append S0's answers to the additional notes
        state['additional_notes'] += f"\nRound {round_num} Notes:\n{s0_additional_notes}"

    return state


def refine_response(original_response, questions, persona_name):
    """Refine the original response based on S0's questions."""
    groq_agent = Groq()
    refinement_prompt = f"""
    You are {persona_name}, refining your response based on the following clarifying questions:
    {questions}

    Original Response: {original_response}

    Provide a refined and updated response based on the questions.
    """

    completion = groq_agent.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": refinement_prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    refined_response = completion.choices[0].message.content.strip()
    print(f"{persona_name}'s Refined Response:\n{refined_response}")
    return refined_response


def generate_embeddings(s1_response, s2_response, s3_response):
    """Generate embeddings for the responses from S1, S2, and S3."""
    embeddings_generator = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Combine the responses
    combined_text = [s1_response, s2_response, s3_response]

    # Generate embeddings for each response
    embeddings_list = embeddings_generator.embed_documents(combined_text)

    # Average the embeddings to produce a single vector
    if embeddings_list and len(embeddings_list) > 0:
        averaged_embeddings = np.mean(embeddings_list, axis=0)  # Average across responses
    else:
        averaged_embeddings = [0.0] * 768  # Default vector of zeros if embeddings fail

    # Convert numpy array to list (which can be inserted into PostgreSQL)
    return averaged_embeddings.tolist()


# Abstract generation using Groq
def abstract_generation(state: ScientistState):
    """Generate the final abstract by summarizing all responses and store embeddings."""
    
    # Combine all responses into one string for the abstract
    combined_responses = f"""
    1. Key insights from S1 (Historical Context): {state['s1_response']}
    2. Additional details provided by S2 (Technical Analysis): {state['s2_response']}
    3. Further perspectives shared by S3 (Ethical Implications): {state['s3_response']}
    """

    # Create the prompt for Groq with all responses
    final_abstract_prompt = f"""
    You are a professional summarizer. Your task is to generate a concise and well-structured abstract by summarizing the below responses:

    {combined_responses}

    Abstract should be between 300-400 words.
    Strictly provide only the abstract, without external information or metadata information.
    """

    # Initialize Groq agent and generate the abstract using Groq
    groq_agent = Groq()
    completion = groq_agent.chat.completions.create(
        model="llama3-8b-8192",  # Use the correct model
        messages=[{"role": "system", "content": final_abstract_prompt}],
        temperature=0.7,
        max_tokens=500,  # Adjust the token length as needed
    )

    # Get the final abstract response
    final_abstract_response = completion.choices[0].message.content.strip()

    # **Generate embeddings** for the final abstract and responses from S1, S2, and S3
    embeddings = generate_embeddings(state['s1_response'], state['s2_response'], state['s3_response'])

    # Store the query response (including embeddings) in the database
    store_query_response(
        topic=state['topic'],
        s1_response=state['s1_response'],
        s2_response=state['s2_response'],
        s3_response=state['s3_response'],
        final_abstract=final_abstract_response,
        embeddings=embeddings  # Pass embeddings generated
    )

    return {'final_abstract': final_abstract_response}



def create_workflow():
    workflow = StateGraph(state_schema=ScientistState)

    # Define nodes
    workflow.add_node("start", start)
    workflow.add_node("query_s0", query_agent_s0)
    workflow.add_node("query_s1", query_agent_s1)
    workflow.add_node("query_s2", query_agent_s2)
    workflow.add_node("query_s3", query_agent_s3)
    workflow.add_node("interview_state", interview_state) 
    workflow.add_node("abstract_generation", abstract_generation)

    # Define edges
    workflow.add_edge(START, "start")
    workflow.add_edge("start", "query_s0")
    workflow.add_edge("query_s0", "query_s1")
    workflow.add_edge("query_s0", "query_s2")
    workflow.add_edge("query_s0", "query_s3")
    workflow.add_edge("query_s1", "interview_state")
    workflow.add_edge("query_s2", "interview_state")
    workflow.add_edge("query_s3", "interview_state")
    workflow.add_edge("interview_state", "abstract_generation")  # Refined responses are sent to final abstract
    workflow.add_edge("abstract_generation", END)

    return workflow