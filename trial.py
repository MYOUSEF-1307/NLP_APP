from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain, LLMChain
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
import json
import os
import time
from tqdm import tqdm
from langchain_together import ChatTogether
# Connect to Neo4j
def create_graph_database():
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    return graph

# Initialize the LLM
def initialize_llm():
    return ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key="d54f037d1f51cff073f2392a83e3b127039f17568d4a50efac44422347c68057"
)

def extract_medication_information(text, llm):
    """Uses LLM to extract structured medication information from text."""
    prompt = f"""
    Extract detailed medication information from the following textbook content and return as valid JSON.
    Focus specifically on medications, their active ingredients, side effects, contraindications, dosages, and mechanisms of action.
    
    Text: "{text}"
    
    Format your response as valid JSON like this example:
    {{
        "medications": [
            {{
                "name": "Aspirin",
                "description": "Non-steroidal anti-inflammatory drug (NSAID) used to treat pain, fever, and inflammation",
                "active_ingredients": ["Acetylsalicylic acid"],
                "side_effects": ["Stomach bleeding", "Heartburn", "Nausea", "Tinnitus"],
                "contraindications": ["Peptic ulcers", "Hemophilia", "Warfarin", "Children under 12"],
                "mechanisms": ["COX-1 and COX-2 inhibition", "Anti-platelet effects"],
                "dosages": ["325-650 mg every 4-6 hours for pain", "81 mg daily for heart attack prevention"],
                "drug_interactions": ["Warfarin", "Other NSAIDs", "Alcohol", "Methotrexate"]
            }}
        ]
    }}
    
    If no medication information is found, return an empty medications array.
    Ensure you only return valid JSON without any additional text.
    """
    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        
        # Clean up response text in case LLM adds any pre or post text around the JSON
        response_text = response_text.strip()
        # Find JSON start and end if there's surrounding text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            response_text = response_text[start_idx:end_idx]
        
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM output: {e}")
        print(f"Response was: {response_text[:200]}...")
        return {"medications": []}
    except Exception as e:
        print(f"Error extracting medication information: {e}")
        return {"medications": []}

def create_medication_graph(data, graph):
    """Inserts extracted medication data into Neo4j."""
    if not data or "medications" not in data or not data["medications"]:
        return False
    
    try:
        # Process medications one at a time
        for medication in data.get("medications", []):
            # Create medication node
            medication_name = medication["name"].replace("'", "\\'")
            description = medication.get("description", "").replace("'", "\\'")
            
            medication_query = f"""
            MERGE (m:Medication {{name: '{medication_name}'}})
            ON CREATE SET m.description = '{description}'
            """
            graph.query(medication_query)
            
            # Process active ingredients
            for ingredient in medication.get("active_ingredients", []):
                ingredient = ingredient.replace("'", "\\'")
                ingredient_query = f"""
                MERGE (i:Ingredient {{name: '{ingredient}'}})
                WITH i
                MATCH (m:Medication {{name: '{medication_name}'}})
                MERGE (m)-[:CONTAINS]->(i)
                """
                graph.query(ingredient_query)
            
            # Process side effects
            for effect in medication.get("side_effects", []):
                effect = effect.replace("'", "\\'")
                effect_query = f"""
                MERGE (s:SideEffect {{name: '{effect}'}})
                WITH s
                MATCH (m:Medication {{name: '{medication_name}'}})
                MERGE (m)-[:HAS_SIDE_EFFECT]->(s)
                """
                graph.query(effect_query)
            
            # Process contraindications
            for contraindication in medication.get("contraindications", []):
                contraindication = contraindication.replace("'", "\\'")
                contraindication_query = f"""
                MERGE (c:Contraindication {{name: '{contraindication}'}})    
                WITH c
                MATCH (m:Medication {{name: '{medication_name}'}})
                MERGE (m)-[:CONTRAINDICATED_WITH]->(c)
                """
                graph.query(contraindication_query)
            
            # Process mechanisms of action
            for mechanism in medication.get("mechanisms", []):
                mechanism = mechanism.replace("'", "\\'")
                mechanism_query = f"""
                MERGE (mech:Mechanism {{name: '{mechanism}'}})
                WITH mech
                MATCH (m:Medication {{name: '{medication_name}'}})
                MERGE (m)-[:WORKS_VIA]->(mech)
                """
                graph.query(mechanism_query)
            
            # Process drug interactions
            for interaction in medication.get("drug_interactions", []):
                interaction = interaction.replace("'", "\\'")
                interaction_query = f"""
                MERGE (i:DrugInteraction {{name: '{interaction}'}})
                WITH i
                MATCH (m:Medication {{name: '{medication_name}'}})
                MERGE (m)-[:INTERACTS_WITH]->(i)
                """
                graph.query(interaction_query)
            
            # Add dosages as properties of the medication
            if "dosages" in medication and medication["dosages"]:
                dosages = "; ".join(medication["dosages"]).replace("'", "\\'")
                dosage_query = f"""
                MATCH (m:Medication {{name: '{medication_name}'}})
                SET m.dosages = '{dosages}'
                """
                graph.query(dosage_query)
        
        return True
    except Exception as e:
        print(f"Error creating medication graph: {e}")
        return False

def load_documents(directory_path):
    """Load documents from a directory supporting multiple file types."""
    loaders = {}
    
    # PDF files
    pdf_loader = DirectoryLoader(
        directory_path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    )
    loaders['pdf'] = pdf_loader
    
    # Text files
    txt_loader = DirectoryLoader(
        directory_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader
    )
    loaders['txt'] = txt_loader
    
    # Add more loaders as needed for other file types
    
    all_docs = []
    for format_name, loader in loaders.items():
        try:
            docs = loader.load()
            print(f"Loaded {len(docs)} {format_name} documents")
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {format_name} documents: {e}")
    
    return all_docs

def chunk_documents(documents, chunk_size=1500, chunk_overlap=200):
    """Split documents into manageable chunks. Increased chunk size for better context."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc.page_content)
        # Add source metadata
        for chunk in doc_chunks:
            chunks.append({"content": chunk, "source": doc.metadata.get("source", "unknown")})
    
    return chunks

def clean_generated_cypher(cypher_text):
    """Clean and fix common issues in generated Cypher queries."""
    # If the query starts with a pattern and not a keyword
    if cypher_text.strip().startswith('('):
        # Prepend MATCH to make it valid
        return "MATCH " + cypher_text
    
    # Check for comma-separated patterns without a proper clause
    if ',' in cypher_text and not any(keyword in cypher_text.upper() for keyword in 
                                      ['MATCH', 'CREATE', 'MERGE', 'RETURN']):
        # Split by commas and create proper MATCH statements
        patterns = cypher_text.split(',')
        cleaned_query = ""
        for i, pattern in enumerate(patterns):
            pattern = pattern.strip()
            if pattern:
                if i == 0:
                    cleaned_query += f"MATCH {pattern}\n"
                else:
                    cleaned_query += f"MATCH {pattern}\n"
        cleaned_query += "RETURN *"
        return cleaned_query
    
    return cypher_text

def format_results(results, question):
    """Format Neo4j results into a readable response."""
    if not results:
        return "I couldn't find any information related to your question."
    
    # Simple formatting for demonstration
    response = f"Here's what I found related to your question about {question}:\n\n"
    
    for i, record in enumerate(results):
        response += f"Result {i+1}:\n"
        for key, value in record.items():
            if isinstance(value, dict) and "name" in value:
                response += f"- {key}: {value['name']}\n"
            else:
                response += f"- {key}: {value}\n"
        response += "\n"
    
    return response

def process_medical_textbooks(books_directory="books", chunk_size=1500, chunk_overlap=200, clear_db=False):
    """Main function to process medical textbooks and build the graph database."""
    print(f"Starting to process medication information from {books_directory}")
    
    # Initialize components
    graph = create_graph_database()
    llm = initialize_llm()
    
    try:
        # Check if database already has content
        node_count = graph.query("MATCH (n) RETURN count(n) as count")[0]["count"]
        
        if node_count > 0:
            print(f"Database already contains {node_count} nodes")
            if clear_db:
                # Only clear if explicitly requested
                graph.query("MATCH (n) DETACH DELETE n")
                print("Cleared existing graph database")
                # Now database is empty, proceed with processing
                should_process_books = True
            else:
                print("Keeping existing database content. Skipping document processing.")
                should_process_books = False
        else:
            print("Database is empty, proceeding with initial build")
            should_process_books = True
        
        # Only process documents if the database is empty or was just cleared
        if should_process_books:
            # Load documents
            print(f"Loading documents from {books_directory}...")
            documents = load_documents(books_directory)
            if not documents:
                print("No documents found. Please check the directory path.")
                return
            
            print(f"Loaded {len(documents)} documents")
            
            # Chunk documents
            print("Chunking documents...")
            chunks = chunk_documents(documents, chunk_size, chunk_overlap)
            print(f"Created {len(chunks)} chunks")
            
            # Process chunks and build graph
            print("Extracting medication information and building graph...")
            successful_chunks = 0
            medication_count = 0
            
            for i, chunk in enumerate(tqdm(chunks)):
                print(f"\nProcessing chunk {i+1}/{len(chunks)} from {chunk['source']}")
                
                # Extract medication information
                data = extract_medication_information(chunk["content"], llm)
                
                # Track how many medications were found
                chunk_medications = len(data.get("medications", []))
                if chunk_medications > 0:
                    print(f"Found {chunk_medications} medications in this chunk")
                    medication_count += chunk_medications
                
                # Create graph from extracted data
                if data and create_medication_graph(data, graph):
                    successful_chunks += 1
                
                # Add a small delay to avoid rate limits with the LLM API
                time.sleep(1)
            
            print(f"\nProcessing complete. Successfully processed {successful_chunks}/{len(chunks)} chunks.")
            print(f"Total medications found: {medication_count}")
            
            # Create indexes for better performance
            indexes = [
                ("Medication", "name"),
                ("Ingredient", "name"),
                ("SideEffect", "name"),
                ("Contraindication", "name"),
                ("Mechanism", "name"),
                ("DrugInteraction", "name")
            ]
            
            for label, property in indexes:
                index_query = f"CREATE INDEX {label}_{property}_index IF NOT EXISTS FOR (n:{label}) ON (n.{property})"
                graph.query(index_query)
        
        # Print some stats about the graph (do this regardless of whether we processed documents)
        medication_count = graph.query("MATCH (m:Medication) RETURN count(m) as count")[0]["count"]
        ingredient_count = graph.query("MATCH (i:Ingredient) RETURN count(i) as count")[0]["count"]
        side_effect_count = graph.query("MATCH (s:SideEffect) RETURN count(s) as count")[0]["count"]
        relationship_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        
        print(f"\nMedication Graph Database Stats:")
        print(f"- Medications: {medication_count}")
        print(f"- Ingredients: {ingredient_count}")
        print(f"- Side Effects: {side_effect_count}")
        print(f"- Total Relationships: {relationship_count}")
        
        # Show top medications by relationship count
        top_medications = graph.query("""
        MATCH (m:Medication)-[r]->()
        WITH m, count(r) as rel_count
        RETURN m.name as medication, rel_count
        ORDER BY rel_count DESC
        LIMIT 10
        """)
        
        print("\nTop 10 Medications by Relationship Count:")
        for record in top_medications:
            print(f"- {record['medication']}: {record['rel_count']} relationships")
        
        # Print visualization instructions
        
        return graph
    
    finally:
        # Ensure the connection is closed properly
        if graph is not None:
            graph._driver.close()
            print("Neo4j connection closed.")

def query_medical_graph(question, graph=None, llm=None):
    """Query the medication knowledge graph using natural language."""
    if not graph:
        graph = create_graph_database()
    
    if not llm:
        llm = initialize_llm()
    
    # Create a custom Cypher generation prompt template
    cypher_generation_template = """
   You are an expert in Neo4j and Cypher query language.
    
    Below is a graph database of medications with the following node types:
    - Medication (medications with properties like name and description)
    - Ingredient (active ingredients in medications)
    - SideEffect (possible side effects of medications)
    - Contraindication (conditions where medications shouldn't be used)
    - Mechanism (mechanisms of action for medications)
    - DrugInteraction (other drugs that interact with medications)
    
    Relationships include:
    - (Medication)-[:CONTAINS]->(Ingredient)
    - (Medication)-[:HAS_SIDE_EFFECT]->(SideEffect)
    - (Medication)-[:CONTRAINDICATED_WITH]->(Contraindication)
    - (Medication)-[:WORKS_VIA]->(Mechanism)
    - (Medication)-[:INTERACTS_WITH]->(DrugInteraction)
    
    Human Question: {question}
    
    Write a valid Cypher query to get all nodes and relationships related to the question.
    Make sure to include the necessary MATCH and RETURN clauses.
    
    Cypher Query:
    """
    
    
    # Create a prompt for Cypher generation
    cypher_prompt = PromptTemplate(
        input_variables=["question"],
        template=cypher_generation_template
    )
    
    # Create a generator for Cypher queries
    cypher_generator = LLMChain(llm=llm, prompt=cypher_prompt)
    
    # Try to query using GraphCypherQAChain first
    try:
        # Create Graph RAG Chain with custom Cypher generator
        qa_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            cypher_llm=llm,  # Use the same LLM
            cypher_prompt=cypher_prompt,  # Use our custom prompt
            allow_dangerous_requests=True
        )
        
        # Query the RAG system
        response = qa_chain.invoke({"query": question})
        return response
    except Exception as e:
        print(f"Error with primary query method: {str(e)}")
        
        # Fall back to direct Cypher generation and execution
        try:
            # Generate Cypher directly
            cypher_response = cypher_generator.invoke({"question": question})
            generated_cypher = cypher_response.get("text", "").strip()
            
            # Clean and fix the generated Cypher
            generated_cypher = clean_generated_cypher(generated_cypher)
            
            print(f"Using fallback method with generated Cypher:\n{generated_cypher}")
            
            # Execute the Cypher query directly
            results = graph.query(generated_cypher)
            
            # Format the results into a response
            formatted_results = format_results(results, question)
            
            return {"result": formatted_results, "query": generated_cypher ,"results":results}
        except Exception as inner_e:
            return {
                "result": f"I couldn't generate a valid query for your question. Error: {str(inner_e)}", 
                "query": ""
            }
    finally:
        # Ensure the connection is closed properly
        if graph is not None and hasattr(graph, '_driver'):
            graph._driver.close()
            print("Neo4j connection closed.")

def answer_with_context(query_results, messages, llm=None):
    """
    Generate a natural language response based on graph query results and conversation history.
    
    Args:
        query_results (dict): The results from query_medical_graph function
        messages (list): List of message dictionaries with 'role' (user/assistant) and 'content'
        llm: The language model to use (will initialize one if not provided)
        
    Returns:
        str: A natural language response addressing the user's query
    """
    if not llm:
        llm = initialize_llm()
    
    # Extract query result information
    if isinstance(query_results, dict) and "result" in query_results:
        graph_info = query_results["result"]
        results= query_results.get("results", [])
        if "query" in query_results:
            cypher_query = query_results.get("query", "")
    else:
        # Handle standard GraphCypherQAChain result format
        graph_info = query_results.get("result", "")
        cypher_query = query_results.get("cypher", "")
    
    # Build conversation history from messages
    conversation_history = ""
    if messages and len(messages) > 0:
        for message in messages:
            role = message.get("role", "").capitalize()
            content = message.get("content", "")
            if role and content:
                conversation_history += f"{role}: {content}\n"
    
    # Create a prompt template for the LLM response
    prompt = f"""
    You are a medical assistant specializing in medication information. Use the following context from a knowledge graph 
    to answer the user's question. Be professional but conversational in your response. if the information is not available, do not say it is lacking the graph info will contain conradicitng drugs
    
    pulled information from the knowledge graph:
    {results}

    KNOWLEDGE GRAPH INFORMATION:
    {graph_info}
    
    PREVIOUS CONVERSATION:
    {conversation_history}
    
    Based on the knowledge graph, provide a helpful and accurate response. keep in mind that the knowledge graph will be relevant to the user's question.
    Focus on answering the latest user question with the information provided, but maintain context from previous messages if relevant.
    If the knowledge graph information is incomplete, acknowledge limitations but provide what you can. Do not show the user that you are missing information , be bold
    Your response:
    """
    print(graph_info)
    print(conversation_history)
    try:
        # Generate response using the LLM
        response = llm.invoke(prompt)
        # Extract the content from the response object
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return "I'm sorry, I couldn't process your question properly. Could you try rephrasing it?"



if __name__ == "__main__":
    # Process medical textbooks and build graph
    books_directory = "books"  # Update this to your books directory path
    graph = process_medical_textbooks(books_directory, clear_db=False)
    llm = initialize_llm()
    
    # Example of using the new function with conversation context
    print("\n=== Demonstrating answer_with_context with conversation history ===")
    
    # Sample conversation history
    conversation = [
        {"role": "user", "content": "i have valvular disease. What medications should I avoid?"},
      
    ]
    
    # Get the latest query from conversation
    latest_query = conversation[-1]["content"]
    
    print(f"\nProcessing query: {latest_query}")
    
    # First, query the knowledge graph
    graph_results = query_medical_graph(latest_query, graph, llm)
    
    # Then, use the answer_with_context function to generate a more conversational response
    contextual_answer = answer_with_context(graph_results, conversation, llm)
    
    print("\n=== Knowledge Graph Raw Result ===")
    if isinstance(graph_results, dict) and "result" in graph_results:
        print(graph_results["result"])
    else:
        print(graph_results)
    
    print("\n=== Contextual Answer Using Conversation History ===")
    print(contextual_answer)
    
