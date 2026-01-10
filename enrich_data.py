import os
import time
import json
import logging
from tqdm import tqdm
from google import genai
from google.genai import types

# ================= CẤU HÌNH (CONFIG) =================
API_KEY = "..."  # <--- THAY API KEY CỦA BẠN VÀO ĐÂY
INPUT_FILE = "..."
OUTPUT_FILE = "..."
BATCH_SIZE = 10
MODEL_NAME = "gemini-2.5-flash"
SLEEP_TIME = 4

# Cấu hình Logging
logging.basicConfig(filename='enrichment.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Khởi tạo Client
client = genai.Client(api_key=API_KEY)

# ================= HÀM TẠO PROMPT (BATCH) =================
def create_batch_prompt(news_batch):
    """
    Tạo prompt xử lý làm giàu cho CẢ Title và Abstract.
    """
    json_input_structure = []
    for news in news_batch:
        json_input_structure.append({
            "id": news['doc_id'],
            "category": news['category'],
            "subcategory": news['subcategory'],
            "original_title": news['title'],
            "original_abstract": news['abstract']
        })
    
    input_str = json.dumps(json_input_structure, ensure_ascii=False)

    prompt = f"""
    Role: You are a meticulous and factual News Editor.
    Task: Enrich BOTH the Title and Abstract of news articles using a hierarchical strategy to improve recommendation quality.

    Input Data (JSON List):
    {input_str}

    For EACH news item, perform these steps internally:

    1. **Direct Rewrite (Clarity & Engagement)**: 
       - Draft a more engaging, click-worthy version of the **Title**.
       - Draft a clearer, more informative version of the **Abstract** (remove noise, fix grammar, make it concise).

    2. **Knowledge Inference (STRICT MODE)**: 
       - Analyze the combined context of the rewritten title and abstract.
       - Identify specific entities (People, Orgs, Locations, Events) implied by the context.
       - **CRITICAL RULE**: Only include entities if they are **FACTUALLY CORRECT** and **DIRECTLY RELEVANT** to the event. 
       - **DO NOT** make up connections. **DO NOT** include entities if you are unsure. It is better to return an empty list than a wrong entity.
       - *Bad Example:* Adding "Elon Musk" to a generic crypto news just because he is famous. (DON'T DO THIS)
       - *Good Example:* Adding "Tesla" to a news about "Model Y recall". (DO THIS)

    3. **Hierarchical Synthesis (Coherent Output)**: 
       - **Final Title**: Combine the engaging style from Step 1 with key entities from Step 2. Max 20 words.
       - **Final Abstract**: Integrate the inferred entities naturally into the abstract text. It should provide full context for the Title. Max 50 words.
       - **Constraint**: The Title and Abstract must be **consistent** with each other and MUST preserve the **EXACT MEANING** of the original news. Do not drift the topic.

    OUTPUT REQUIREMENT:
    Return a JSON LIST of objects with exactly these fields:
    - "id": The original doc_id.
    - "enriched_title": The final title from Step 3.
    - "enriched_abstract": The final abstract from Step 3.
    """
    return prompt

# ================= HÀM XỬ LÝ CHÍNH =================
def process_pipeline():
    abs_output_path = os.path.abspath(OUTPUT_FILE)
    print(f"=== BẮT ĐẦU PIPELINE ===")
    print(f"File đầu ra sẽ nằm tại: {abs_output_path}")
    
    # 1. Kiểm tra Resume
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"Phát hiện file {OUTPUT_FILE}, đang kiểm tra dữ liệu đã xử lý để resume...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > 0:
                    processed_ids.add(parts[0])
        print(f"--> Đã tìm thấy {len(processed_ids)} bài báo đã xử lý. Sẽ bỏ qua chúng.")

    # 2. Đọc dữ liệu đầu vào
    print("Đang đọc file input...")
    all_news = []
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5: continue
                
                doc_id = parts[0]
                if doc_id in processed_ids: continue

                all_news.append({
                    "doc_id": parts[0],
                    "category": parts[1],
                    "subcategory": parts[2],
                    "title": parts[3],
                    "abstract": parts[4],
                    "url": parts[5] if len(parts) > 5 else "",
                    "raw_line": parts
                })
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file input tại {INPUT_FILE}")
        return
    
    total_news = len(all_news)
    print(f"--> Cần xử lý: {total_news} bài báo.")
    
    if total_news == 0:
        print("Tất cả đã hoàn thành! Không còn gì để làm.")
        return

    # 3. Batch Processing Loop
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        
        for i in tqdm(range(0, total_news, BATCH_SIZE), desc="Enriching Batches"):
            batch = all_news[i : i + BATCH_SIZE]
            
            prompt = create_batch_prompt(batch)
            result_mapping = {} # Map id -> {title, abstract}
            
            try:
                # --- GỌI API ---
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2, # Giữ thấp để tránh bịa đặt
                        top_p=0.8,
                        top_k=40
                    )
                )
                
                # Parse JSON trả về
                if response.text:
                    json_data = json.loads(response.text)
                    for item in json_data:
                        # Lấy cả Title và Abstract mới
                        if "id" in item:
                            result_mapping[item["id"]] = {
                                "title": item.get("enriched_title", ""),
                                "abstract": item.get("enriched_abstract", "")
                            }
                else:
                    logging.warning(f"Batch {i}: API trả về text rỗng.")

            except Exception as e:
                logging.error(f"Batch Error tại index {i}: {str(e)}")
                print(f"\n[Warning] Batch lỗi: {e}. Dùng data gốc cho batch này.")
                time.sleep(10)

            # Ghi kết quả
            for news_item in batch:
                doc_id = news_item['doc_id']
                original_parts = news_item['raw_line']
                
                # Lấy kết quả từ API (hoặc fallback về gốc nếu lỗi/thiếu)
                enrich_data = result_mapping.get(doc_id, {})
                
                new_title = enrich_data.get("title", news_item['title'])
                new_abstract = enrich_data.get("abstract", news_item['abstract'])
                
                # Fallback an toàn nếu chuỗi rỗng
                if not new_title: new_title = news_item['title']
                if not new_abstract: new_abstract = news_item['abstract']
                
                # Làm sạch chuỗi (xóa tab/newline để không vỡ format TSV)
                new_title = new_title.replace('\t', ' ').replace('\n', ' ').strip()
                new_abstract = new_abstract.replace('\t', ' ').replace('\n', ' ').strip()
                
                # CẬP NHẬT VÀO LIST GỐC
                # MIND Format: [ID(0), Cat(1), SubCat(2), Title(3), Abstract(4), ...]
                if len(original_parts) > 4:
                    original_parts[3] = new_title      # Cập nhật Title
                    original_parts[4] = new_abstract   # Cập nhật Abstract
                
                output_line = '\t'.join(original_parts) + '\n'
                f_out.write(output_line)
            
            f_out.flush()
            time.sleep(SLEEP_TIME)

    print(f"\nHoàn thành! File đã lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_pipeline()
