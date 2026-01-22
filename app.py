import streamlit as st
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from google import genai
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Disaster Response AI", layout="wide")
st.title("🚨 Crisis Intelligence Command Center")

# --- Caching for speed ---
@st.cache_resource # this is critical because Streamlit reruns the entire script every time a user interacts with a button or slider. Without caching, your app would try to reconnect to Qdrant or reload your AI models on every single click, making it extremely slow.
def load_models():
    # Text Encoder (Day 3)
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # CLIP for Images (Day 2)
    clip_model = SentenceTransformer('clip-ViT-B-32')
    # Clients
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_KEY = os.getenv("QDRANT_API_KEY")
    q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    g_client = genai.Client(api_key=GEMINI_KEY)


    collections_to_init = {
        "user_episodic_memory": 384, 
        "disaster_multimodal": 512
    }
    
    for name, size in collections_to_init.items():
        # Check if collection exists before creating
        existing = q_client.get_collections()
        if not any(c.name == name for c in existing.collections):
            q_client.create_collection(
                collection_name=name,
                vectors_config={"size": size, "distance": "Cosine"}
            )

    return text_encoder, clip_model, q_client, g_client

text_encoder, clip_model, q_client, g_client = load_models()

# --- SIDEBAR: SYSTEM MEMORY ---
with st.sidebar:
    st.header("🧠 System Memory")
    st.info("This panel shows what the AI is currently retrieving from Qdrant.")
    memory_placeholder = st.empty()

    if st.button("🔄 Start New Scenario"):
        # 1. Clear ONLY User/Assistant logs
        # This keeps 'system_report' (Base Data) and all Images safe
        try:
            q_client.delete(
                collection_name="user_episodic_memory",
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="role",
                                match=models.MatchAny(any=["user", "assistant"])
                            )
                        ]
                    )
                )
            )
            st.success("Short-term memory wiped!")
        except Exception as e:
            st.error(f"Error clearing memory: {e}")
    
        # 2. Clear Streamlit Chat History
        st.session_state.messages = []
    
        # 3. Force Refresh
        st.rerun()

# --- MAIN INTERFACE: CHAT ---

# working explanation: 
# st.session_state: This is a specialized dictionary provided by Streamlit that persists across reruns for as long as the user's browser tab is open.

# if "messages" not in ...: This is a safety check. It asks: "Is there already a conversation history started in this session?"

# st.session_state.messages = []: If the answer is "no" (which only happens the very first time the page loads), it creates an empty list to store the chat history.

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. DISPLAY HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # PERSISTENT IMAGE & REASONING
        if message.get("image"):
            title = f"Retrieved Evidence (Score: {message.get('score', 0.0):.2f})"
            st.image(message["image"], caption=title, use_container_width=True)
            
            if message.get("reasoning"):
                st.caption(f"Reasoning: Found image matching description '{message['reasoning']}'")

        # PERSISTENT SOURCES EXPANDER
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("🔍 See Evidence & Sources"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Text Memories:**")
                    for msg in message["sources"]: st.caption(f"• {msg}")
                with col2:
                    if message.get("image"):
                        st.image(message["image"], width=150)
                        st.caption(f"Score: {message.get('score', 0):.2f}")

# --- CHAT LOGIC ---
if prompt := st.chat_input("Ask about a disaster scenario..."):

    negative_keywords = ["don't", "dont", "no photo", "no image", "stop showing"]
    user_wants_no_image = any(word in prompt.lower() for word in negative_keywords)

    # 1. Display User Message
    # understand working
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. RETRIEVAL 
    # Search Memory
    try: 
        query_vec = text_encoder.encode(prompt).tolist()
        mem_hits = q_client.query_points(collection_name="user_episodic_memory", query=query_vec, limit=2).points
        context_list = [
        h.payload['chat_text'] 
        for h in mem_hits 
        if h.score > 0.40  # <--- THE NOISE FILTER
    ]
    except Exception as e:
        # If collection missing (404), treat it as an empty context
        context_list = []
    
    context = "\n".join(context_list) if context_list else "NO RELEVANT DATA FOUND IN DATABASE."

    # Search Images (CLIP)
    try:
        img_vec = clip_model.encode(prompt).tolist()
        img_hits = q_client.query_points(collection_name="disaster_multimodal", query=img_vec, limit=1).points
    except Exception as e:
        # If collection is missing, ensure no image is triggered
        img_hits = []

    visual_context = "No relevant images found."
    if img_hits and img_hits[0].score > 0.25:
        visual_context = f"A relevant image was found showing: {img_hits[0].payload['description']}"

    # 3. GENERATION
    full_prompt = f"""
    You are a Disaster Response Assistant.

    PAST TEXT LOGS: 
    {context}

    VISUAL EVIDENCE FOUND: 
    {visual_context}

    INSTRUCTIONS:
    1. Use BOTH the text logs and the visual evidence to answer.
    2. If the DATABASE CONTEXT is 'NO RELEVANT DATA FOUND', inform the user you have no specific local logs on this, but provide helpful GENERAL advice based on your own knowledge.
    3. If an image was found but no text was found, describe the image to the user as your primary evidence.
    4. If the user asks for a photo and none is found, simply state you have no visual evidence available.
    5. Be professional and concise.

    USER QUESTION: {prompt}
    """
    response = g_client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
    
    # 4. Display Assistant Response
    with st.chat_message("assistant"):
        st.markdown(response.text)
        
        # Show retrieved image if relevant
        found_image = None
        img_score = 0
        reasoning = ""  
        if img_hits and img_hits[0].score > 0.25 and not user_wants_no_image:
            found_image = img_hits[0].payload['filename']
            img_score = img_hits[0].score
            reasoning = f"{img_hits[0].payload['description']}"
            st.image(img_hits[0].payload['filename'], caption=f"Retrieved Evidence (Score: {img_hits[0].score:.2f})")
            st.caption(f"Reasoning: Found image matching description '{img_hits[0].payload['description']}'")
        
        elif user_wants_no_image:
            st.info("Image display suppressed by user request.")

        # NEW: Expandable Source Panel
        with st.expander("🔍 See Evidence & Sources"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Text Memories Used:**")
                if context_list:
                    for msg in context_list:
                        st.caption(f"• {msg}")
                else:
                    st.caption("No relevant past memories found.")
            
            with col2:
                st.write("**Visual Evidence:**")
                if img_hits and img_hits[0].score > 0.25:
                    st.image(img_hits[0].payload['filename'], width=150)
                    st.caption(f"Match Score: {img_hits[0].score:.2f}")
                else:
                    st.caption("No matching images found.") 
    # Update Sidebar with the "thought process"
    memory_placeholder.write(f"**Retrieved Context:**\n{context}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.text,
        "image": found_image,
        "caption": "Retrieved Evidence" if found_image else None,
        "reasoning": reasoning,
        "sources": context_list,
        "score": img_score
    })
    # 1. Store the User's question
    q_client.upsert(
        collection_name="user_episodic_memory",
        points=[{
            "id": str(uuid.uuid4()),
            "vector": text_encoder.encode(prompt).tolist(),
            "payload": {"chat_text": prompt, "role": "user"}
        }]
    )

    # 2. Store the AI's response
    q_client.upsert(
        collection_name="user_episodic_memory",
        points=[{
            "id": str(uuid.uuid4()),
            "vector": text_encoder.encode(response.text).tolist(),
            "payload": {"chat_text": response.text, "role": "assistant"}
        }]
    )
