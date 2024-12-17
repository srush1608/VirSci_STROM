import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from workflow import generate_embeddings
import os

# Load environment variables
load_dotenv()

# Database connection credentials
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def connect_db():
    """
    Establishes a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print("Error connecting to the database:", e)
        return None


def create_table():
    """
    Creates the necessary table to store query and response data if it doesn't exist.
    """
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()

            # Enable pgvector extension for vector storage
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create the table with all necessary columns if not already created
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_chat_history (
                    id SERIAL PRIMARY KEY,
                    topic VARCHAR(255),
                    s1_response TEXT,
                    s2_response TEXT,
                    s3_response TEXT,
                    final_abstract TEXT,
                    embeddings VECTOR(768),  -- Embedding size for the abstract, assuming 768 dimensions
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            print("Table 'research_chat_history' created successfully.")
        except Exception as e:
            print("Error creating table:", e)
        finally:
            cursor.close()
            conn.close()


def store_query_response(topic, s1_response, s2_response, s3_response, final_abstract, embeddings):
    """
    Stores the query (topic) and responses (S1, S2, S3), the final abstract, and embeddings in the database.
    """
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO research_chat_history 
                (topic, s1_response, s2_response, s3_response, final_abstract, embeddings)
                VALUES (%s, %s, %s, %s, %s, %s);
            """
            # Ensure embeddings are passed as a list (already converted in generate_embeddings)
            cursor.execute(query, (topic, s1_response, s2_response, s3_response, final_abstract, embeddings))
            conn.commit()
            print(f"Data for topic '{topic}' stored successfully.")
        except Exception as e:
            print("Error inserting data into table:", e)
        finally:
            cursor.close()
            conn.close()




def get_all_results():
    """
    Retrieves all stored research results.
    Returns a list of tuples: [(id, topic, s1_response, s2_response, s3_response, final_abstract, timestamp), ...]
    """
    conn = connect_db()
    results = []
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, topic, s1_response, s2_response, s3_response, final_abstract, timestamp
                FROM research_chat_history
                ORDER BY timestamp DESC;
            """)
            results = cursor.fetchall()
        except Exception as e:
            print("Error retrieving all data:", e)
        finally:
            cursor.close()
            conn.close()
    return results


def get_result_by_topic(topic):
    """
    Retrieves a specific research result by topic.
    Returns a single row as a tuple: (id, topic, s1_response, s2_response, s3_response, final_abstract, embeddings, timestamp)
    """
    conn = connect_db()
    result = None
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, topic, s1_response, s2_response, s3_response, final_abstract, embeddings, timestamp
                FROM research_chat_history
                WHERE topic = %s;
            """, (topic,))
            result = cursor.fetchone()
        except Exception as e:
            print(f"Error retrieving data for topic '{topic}':", e)
        finally:
            cursor.close()
            conn.close()
    return result


def update_final_abstract(topic, final_abstract, embeddings):
    """
    Updates the final abstract and embeddings for a specific topic.
    """
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            query = """
                UPDATE research_chat_history
                SET final_abstract = %s, embeddings = %s, timestamp = CURRENT_TIMESTAMP
                WHERE topic = %s;
            """
            cursor.execute(query, (final_abstract, embeddings, topic))
            conn.commit()
            print(f"Record for topic '{topic}' updated successfully.")
        except Exception as e:
            print(f"Error updating record for topic '{topic}':", e)
        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    # Initialize the database and create the table
    create_table()
    print("Database initialized and ready.")
