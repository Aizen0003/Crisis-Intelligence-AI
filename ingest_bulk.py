import os
import uuid
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from PIL import Image
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
q_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = SentenceTransformer('clip-ViT-B-32')

IMAGE_FOLDER = "data_images" # Put your 60 images here
TEXT_FILE = "data_logs.txt"  # Put your 50 text lines here

def bulk_ingest():
    print("🚀 Starting Bulk Ingestion...")

    # --- PART A: IMAGES ---
    if os.path.exists(IMAGE_FOLDER):
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"📸 Found {len(image_files)} images. Processing...")
        
        img_points = []
        for img_file in image_files:
            try:
                # 1. Generate Description from Filename
                # "guwahati_flood_zoo_road.jpg" -> "guwahati flood zoo road"
                clean_desc = img_file.split('.')[0].replace('_', ' ').replace('-', ' ')
                
                # 2. Encode
                img_path = os.path.join(IMAGE_FOLDER, img_file)
                vector = clip_model.encode(Image.open(img_path)).tolist()
                
                # 3. Prepare Point
                img_points.append({
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "filename": img_path,
                        "description": clean_desc,
                        "type": "photo"
                    }
                })
            except Exception as e:
                print(f"⚠️ Skipped {img_file}: {e}")
        
        # Upload all images at once (Batched)
        if img_points:
            q_client.upsert(collection_name="disaster_multimodal", points=img_points)
            print(f"✅ Successfully uploaded {len(img_points)} images.")
    else:
        print(f"❌ Folder '{IMAGE_FOLDER}' not found!")

    # --- PART B: TEXT LOGS ---
    if os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, 'r') as f:
            text_lines = [line.strip() for line in f if line.strip()]
        
        print(f"📄 Found {len(text_lines)} text logs. Processing...")
        
        txt_points = []
        for text in text_lines:
            txt_points.append({
                "id": str(uuid.uuid4()),
                "vector": text_encoder.encode(text).tolist(),
                "payload": {"chat_text": text, "role": "system_report"}
            })
            
        if txt_points:
            q_client.upsert(collection_name="user_episodic_memory", points=txt_points)
            print(f"✅ Successfully uploaded {len(txt_points)} text logs.")
    else:
        print(f"❌ File '{TEXT_FILE}' not found!")

if __name__ == "__main__":
    bulk_ingest()