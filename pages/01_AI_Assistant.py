import streamlit as st
import openai
import os
from typing import List
from datetime import datetime
from tempfile import gettempdir
from PIL import Image
from io import BytesIO
import warnings
import logging
from openai.types.beta.assistant_stream_event import ThreadMessageDelta
from openai.types.beta.threads.text_delta_block import TextDeltaBlock
from keboola_streamlit import KeboolaStreamlit

# Filter out the specific deprecation warning
warnings.filterwarnings("ignore", message=".*_get_websocket_headers.*")
# Also suppress StreamlitAPIWarning logs
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

# Set page config
st.set_page_config(
    page_title="Call Transcript AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Only use valid deprecation settings
if hasattr(st, 'set_option'):
    try:
        st.set_option('deprecation.showWarningOnDirectUse', False)
    except:
        pass

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
keboola = KeboolaStreamlit(st.secrets["kbc_url"], st.secrets["kbc_token"])

session_defaults = {
    "messages": [],
    "file_ids": [],
    "file_ids_df": None
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message .message-content {
        display: flex;
        margin-bottom: 0.5rem;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .chat-message.user .avatar {
        background-color: #f0f2f6;
        color: white;
    }
    .chat-message .content {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<h1 class='main-header'>🤖 Call Transcript AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("""
This AI assistant can help you analyze call transcript data and provide insights. 
Ask questions about customer interactions, sentiment analysis, common topics, 
agent performance, or suggestions for improvements.
""")

# Sidebar with logo
try:
    from PIL import Image
    import os
    
    logo_path = "/data/in/files/kbl.png"
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        st.sidebar.image(logo, use_container_width=True)
    else:
        st.sidebar.info("Keboola logo not found at the expected path.")
except Exception as e:
    st.sidebar.info(f"Could not load logo: {e}")

# Sidebar controls
st.sidebar.title("Assistant Controls")
if st.sidebar.button("Clear Chat History", type="primary"):
    st.session_state.messages = []
    if "thread_id" in st.session_state:
        del st.session_state.thread_id
    st.rerun()

def get_file_ids_from_csv() -> List[str]:
    """Read file IDs from a CSV file."""
    if st.session_state.file_ids_df is None:
        with st.spinner("Loading data..."):
            st.session_state.file_ids_df = keboola.read_table(st.secrets["file_upload_data_app"])
        
        # Create a mapping of file_id to file_name for easier reference
        if "file_mapping" not in st.session_state:
            st.session_state.file_mapping = {
                row['file_id']: row['file_name'] 
                for _, row in st.session_state.file_ids_df.iterrows()
            }
    
    return st.session_state.file_ids_df['file_id'].tolist()

def initialize_assistant() -> str:
    """Initialize or retrieve the assistant ID."""
    if "assistant_id" not in st.session_state:
        st.session_state.assistant_id = st.secrets["ASSISTANT_ID"]
    return st.session_state.assistant_id

def create_thread(file_ids: List[str]) -> str:
    """Create a new thread or retrieve existing thread ID."""
    if "thread_id" not in st.session_state:
        attachments = [{"file_id": file_id, "tools": [{"type": "code_interpreter"}]} for file_id in file_ids]
        
        # Include file mapping information in the initial message
        file_info = "\n".join([
            f"- {st.session_state.file_mapping[file_id]}: {file_id}" 
            for file_id in file_ids if file_id in st.session_state.file_mapping
        ])
        
        initial_content = f"""
        Current date is {datetime.now().strftime('%B %d, %Y')}.
        
        You are an assistant specialized in analyzing call transcript data. You can help with:
        1. Finding patterns in customer interactions
        2. Analyzing sentiment across calls
        3. Identifying common topics or issues
        4. Providing insights on agent performance
        5. Suggesting improvements for call handling
        
        Available files:
        {file_info}
        """
        
        try:
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": initial_content,
                        "attachments": attachments
                    }
                ]
            )
            
            st.session_state.thread_id = thread.id
        except Exception as e:
            st.error(f"Error creating thread: {e}")
            st.info("Please check your OpenAI API key and assistant ID in secrets.")
            return None
    
    return st.session_state.thread_id

# Initialize assistant and thread
with st.spinner("Initializing AI Assistant..."):
    assistant_id = initialize_assistant()
    st.session_state.file_ids = get_file_ids_from_csv()
    thread_id = create_thread(st.session_state.file_ids)

if not thread_id:
    st.stop()

# Create a container for the chat
chat_container = st.container()

# Display previous messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            avatar = '👤'
        else:
            avatar = 'https://components.keboola.com/images/default-app-icon.png'
        
        with st.chat_message(message["role"], avatar=avatar):
            if "[Image:" in message["content"]:
                start_index = message["content"].find("[Image:") + len("[Image: ")
                end_index = message["content"].find("]", start_index)
                image_path = message["content"][start_index:end_index]
                st.image(image_path)
                text_content = message["content"][:start_index - len("[Image: ")] + message["content"][end_index + 1:]
                st.markdown(text_content)
            else:
                st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask me about the call transcript data...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar='https://components.keboola.com/images/default-app-icon.png'):
            # Create a placeholder for the assistant's response
            message_placeholder = st.empty()
            assistant_reply = ""
            
            try:
                # Create the message in the thread
                thread_message = client.beta.threads.messages.create(
                    st.session_state.thread_id,
                    role="user",
                    content=prompt,
                )
                
                # Create a streaming run
                stream = client.beta.threads.runs.create(
                    thread_id=st.session_state.thread_id,
                    assistant_id=assistant_id,
                    stream=True
                )
                
                # Process the streaming response
                for event in stream:
                    if isinstance(event, ThreadMessageDelta):
                        if isinstance(event.data.delta.content[0], TextDeltaBlock):
                            # Clear the placeholder
                            message_placeholder.empty()
                            
                            # Add the new text
                            assistant_reply += event.data.delta.content[0].text.value
                            
                            # Display the updated text
                            message_placeholder.markdown(assistant_reply)
                
                # After streaming is complete, process any images
                messages = client.beta.threads.messages.list(
                    thread_id=st.session_state.thread_id
                )
                
                newest_message = messages.data[0]
                complete_message_content = assistant_reply
                
                for message_content in newest_message.content:
                    if hasattr(message_content, "image_file"):
                        file_id = message_content.image_file.file_id
                        resp = client.files.with_raw_response.retrieve_content(file_id)
                        if resp.status_code == 200:
                            image_data = BytesIO(resp.content)
                            img = Image.open(image_data)
                            temp_dir = gettempdir()
                            image_path = os.path.join(temp_dir, f"{file_id}.png")
                            img.save(image_path)
                            st.image(img)
                            complete_message_content += f"[Image: {image_path}]\n"
                
                # Store the complete message in session state
                st.session_state.messages.append({"role": "assistant", "content": complete_message_content})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                message_placeholder.markdown("I'm sorry, I encountered an error while processing your request. Please try again.")

# Sidebar info section with example questions
st.sidebar.markdown("---")
st.sidebar.header("Example Questions")
example_questions = [
    "What are the most common topics in customer calls?",
    "Analyze sentiment trends across different call types",
    "What percentage of calls have negative sentiment?",
    "Which agents have the highest customer satisfaction?",
    "What are the most common reasons for customers to call?"
]

for question in example_questions:
    if st.sidebar.button(question, key=f"example_{hash(question)}"):
        # This simulates clicking the question into the chat input
        st.session_state.messages.append({"role": "user", "content": question})
        st.rerun()
