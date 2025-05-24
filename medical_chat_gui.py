import gradio as gr
import datetime
import tempfile
import os
import whisper
from trial import initialize_llm, create_graph_database, query_medical_graph, answer_with_context

# Initialize components
def initialize_components():
    print("Initializing LLM and graph database...")
    llm = initialize_llm()
    graph = create_graph_database()
    return llm, graph

# Initialize Whisper model for speech-to-text
def initialize_whisper_model():
    print("Loading Whisper model for speech recognition...")
    # Use "base" model for a good balance between accuracy and speed
    # You can change to "tiny", "small", "medium", or "large" based on your needs
    model = whisper.load_model("small")
    return model

# Global variables
llm, graph = initialize_components()
whisper_model = initialize_whisper_model()
conversation_history = []
user_knowledge = ""
is_first_message = True

# Speech to text conversion using Whisper
def transcribe_audio(audio_file):
    """
    Transcribe speech from audio file using Whisper
    """
    if audio_file is None:
        return ""
    
    try:
        # Get file path from Gradio's audio component
        if isinstance(audio_file, tuple):
            # If audio_file is a tuple (sample_rate, data), save it to a temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.close()
            
            # Save the recorded audio to the temp file
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_file.name, audio_file[0], audio_file[1])
            audio_path = temp_file.name
        else:
            # If it's already a file path
            audio_path = audio_file
        
        # Transcribe audio using Whisper
        print(f"Transcribing audio file: {audio_path}")
        result = whisper_model.transcribe(audio_path,language="en")
        transcription = result["text"].strip()
        
        # Clean up temp file if we created one
        if isinstance(audio_file, tuple) and os.path.exists(audio_path):
            os.unlink(audio_path)
            
        print(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return f"Error transcribing audio: {str(e)}"

# Preprocess user query with LLM to better understand intent
def preprocess_query_with_llm(query, conversation_history, llm):
    """
    Use the LLM to understand and enhance the user's query based on conversation history
    before sending it to the knowledge graph.
    """
    # If this is the first message or conversation is very short, just return the original query
    if len(conversation_history) <= 1:
        return query, "No preprocessing needed for first message"
    
    # Format the conversation history for the prompt
    formatted_history = ""
    for message in conversation_history[-4:]:  # Include up to last 4 messages for context
        role = message.get("role", "").capitalize()
        content = message.get("content", "")
        formatted_history += f"{role}: {content}\n"
    
    # Create a prompt for the LLM to analyze and enhance the query
    prompt = f"""
    You are an AI that helps understand and enhance user queries for a medical knowledge graph.
    
    Given the conversation history and the user's latest query, your job is to:
    1. Understand what the user is asking in the context of the conversation
    2. Rewrite the query to be clear, specific, and include any relevant context from the conversation
    3. Make sure important medical terms, medication names, or conditions from the conversation are included
    
    CONVERSATION HISTORY:
    {formatted_history}
    
    LATEST QUERY: {query}
    
    Enhanced query: 
    """
    
    try:
        # Get the enhanced query from the LLM
        response = llm.invoke(prompt)
        enhanced_query = response.content if hasattr(response, "content") else str(response)
        
        # Clean up the response
        enhanced_query = enhanced_query.strip()
        
        # Log the transformation
        transformation_log = f"Original: '{query}'\nEnhanced: '{enhanced_query}'"
        print(f"Query transformed:\n{transformation_log}")
        
        return enhanced_query, transformation_log
    except Exception as e:
        print(f"Error in query preprocessing: {e}")
        return query, f"Error in preprocessing: {str(e)}"

# Process query and generate response
def process_query(query, knowledge_text, chat_history):
    global conversation_history, user_knowledge, is_first_message

    # Skip empty queries
    if not query.strip():
        return "", chat_history

    # Update user knowledge if needed
    if knowledge_text.strip():
        user_knowledge = knowledge_text.strip()

    # Add the current query to our internal conversation history
    conversation_history.append({"role": "user", "content": query})

    # First, preprocess the query using the LLM
    enhanced_query, transform_log = preprocess_query_with_llm(query, conversation_history, llm)

    # Check if this is the first message and we have user knowledge to include
    first_message_with_knowledge = is_first_message and user_knowledge
    if first_message_with_knowledge:
        # Create an enhanced query that includes user knowledge for the first message
        final_query = f"User query: {enhanced_query}\nUser background: {user_knowledge}"
        query_response = query_medical_graph(final_query, graph, llm)
    else:
        # Use the enhanced query
        query_response = query_medical_graph(enhanced_query, graph, llm)
    
    # Get a contextual response based on the query result and full conversation history
    response = answer_with_context(query_response, conversation_history, llm)
    
    # Add the response to our internal conversation history
    conversation_history.append({"role": "assistant", "content": response})
    
    # Mark that we've processed the first message
    if is_first_message:
        is_first_message = False
    
    # Format for Gradio's chat interface (list of tuples)
    chat_history.append((query, response))
    return "", chat_history

# Reset the conversation
def reset_conversation():
    global conversation_history, is_first_message
    conversation_history = []
    is_first_message = True
    return [], ""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Medical Knowledge Graph Assistant") as demo:
        gr.Markdown("# Medical Knowledge Graph Assistant")
        gr.Markdown("Ask questions about medications, side effects, interactions, and more. You can type or speak your questions.")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False,
                    show_copy_button=True
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about medications, side effects, interactions, and more...",
                        show_label=False,
                        container=False
                    )
                    submit_btn = gr.Button("Send", variant="primary")
                
                # Add voice input capabilities
                gr.Markdown("### Voice Input")
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone"], 
                        type="numpy", 
                        label="Record your question"
                    )
                    transcribe_btn = gr.Button("Transcribe", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Knowledge About User")
                gr.Markdown("This information will be included with your first query only.")
                user_info = gr.Textbox(
                    placeholder="Enter relevant medical history, conditions, allergies, or other context about the user...",
                    lines=10,
                    label="",
                    info="This information is included only in the first message"
                )
                reset_btn = gr.Button("Start New Conversation", variant="secondary")
        
        # Set up interactions
        submit_btn.click(
            process_query, 
            [msg, user_info, chatbot], 
            [msg, chatbot]
        )
        msg.submit(
            process_query, 
            [msg, user_info, chatbot], 
            [msg, chatbot]
        )
        
        # Set up voice input processing
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[msg]
        )
        
        reset_btn.click(
            reset_conversation,
            outputs=[chatbot, user_info]
        )
        
        gr.Markdown(f"### Current Date: {datetime.datetime.now().strftime('%B %d, %Y')}")
        gr.Markdown("*Note: This application provides information only, not medical advice. Always consult a healthcare provider.*")
    
    return demo

# Main execution
if __name__ == "__main__":
    demo = create_interface()
    # Launch the interface
    demo.launch(share=False)