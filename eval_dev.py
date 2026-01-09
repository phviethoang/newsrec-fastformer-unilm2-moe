import os
import torch
import numpy as np
import logging
from parameters import parse_args
from utility.utils import setuplogger, get_device
from utility.metrics import MetricsDict
from data_handler.preprocess import get_news_feature, infer_news
from data_handler.TestDataloader import DataLoaderTest
from models.speedyrec import MLNR

# ==============================================================================
# HÀM TEST ĐỘC LẬP (Tách từ code cũ ra để chạy lẻ)
# ==============================================================================
def run_evaluation(args):
    setuplogger()
    device = get_device()
    
    # 1. Đường dẫn Checkpoint cần test
    # (Bạn có thể truyền qua args.load_ckpt_name hoặc sửa cứng ở đây nếu lười gõ lệnh)
    if not args.load_ckpt_name:
        raise ValueError("Vui lòng cung cấp đường dẫn checkpoint qua tham số --load_ckpt_name")
        
    logging.info(f"--> Đang load checkpoint từ: {args.load_ckpt_name}")
    
    # 2. Khởi tạo Model & Load Weight
    # Lưu ý: Cần load checkpoint trước để lấy category_dict nếu model phụ thuộc vào nó
    # Nhưng trong SpeedyRec, model khởi tạo trước rồi mới load state dict.
    
    checkpoint = torch.load(args.load_ckpt_name, map_location='cpu')
    
    # Lấy lại dict từ checkpoint để đảm bảo khớp với lúc train (QUAN TRỌNG)
    category_dict = checkpoint.get('category_dict', None)
    subcategory_dict = checkpoint.get('subcategory_dict', None)
    
    # Khởi tạo model
    model = MLNR(args)
    model.to(device)
    
    # Load trọng số
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info("--> Load Model thành công!")

    # 3. Chuẩn bị dữ liệu News (Enriched)
    # Hàm này sẽ đọc file news.tsv trong folder dev/ (đã được bạn enrich và merge)
    logging.info("--> Đang encode toàn bộ News trong tập Dev...")
    with torch.no_grad():
        # mode='dev' sẽ trỏ vào folder args.root_data_dir/dev/
        news_info, news_combined = get_news_feature(
            args, 
            mode='dev', 
            category_dict=category_dict, 
            subcategory_dict=subcategory_dict
        )
        
        # Encode tất cả tin tức thành vector
        news_vecs = infer_news(model, device, news_combined)
        logging.info(f"--> Đã encode xong {news_vecs.shape[0]} bài báo.")

    # 4. Chạy DataLoader Test
    # Nó sẽ đọc file behaviors.tsv trong folder dev/ và ghép với news_vecs ở trên
    logging.info("--> Bắt đầu chạy đánh giá (Inference)...")
    dataloader = DataLoaderTest(
        news_index=news_info.news_index,
        news_scoring=news_vecs,
        data_dirs=[os.path.join(args.root_data_dir, 'dev/')],
        filename_pat=args.filename_pat,
        args=args,
        world_size=1,
        worker_rank=0,
        cuda_device_idx=0,
        enable_prefetch=args.enable_prefetch,
        enable_shuffle=False, # Test thì không cần shuffle
        enable_gpu=args.enable_gpu,
    )

    # 5. Tính toán Metrics
    results = MetricsDict(metrics_name=["AUC", "MRR", "nDCG5", "nDCG10"])
    results.add_metric_dict('all users')
    
    with torch.no_grad():
        for cnt, (log_vecs, log_mask, news_vecs_batch, labels) in enumerate(dataloader):
            
            if args.enable_gpu:
                log_vecs = log_vecs.cuda(device=device, non_blocking=True)
                log_mask = log_mask.cuda(device=device, non_blocking=True)

            # User Encoder: Tạo vector user từ lịch sử
            user_vecs = model.user_encoder(
                log_vecs, log_mask, user_log_mask=True
            ).cpu().detach().numpy()

            # Tính điểm (Dot Product)
            for user_vec, news_vec, label in zip(user_vecs, news_vecs_batch, labels):
                if label.mean() == 0 or label.mean() == 1:
                    continue
                
                score = np.dot(news_vec, user_vec)
                
                # Cập nhật kết quả
                metric_rslt = results.cal_metrics(score, label)
                results.update_metric_dict('all users', metric_rslt)
            
            if cnt % 100 == 0:
                print(f"Processed {cnt} batches...", end='\r')

    # 6. In kết quả cuối cùng
    logging.info("\n" + "="*30)
    logging.info(f"KẾT QUẢ TEST TRÊN TẬP DEV (Enriched)")
    logging.info("="*30)
    results.print_metrics(0, 0, 'all users')
    logging.info("="*30)

if __name__ == '__main__':
    args = parse_args()
    # Ép một số tham số để chạy test mode chuẩn
    args.world_size = 1
    run_evaluation(args)

# import os
# import torch
# import numpy as np
# import logging
# from parameters import parse_args
# from utility.utils import setuplogger, get_device
# from utility.metrics import MetricsDict
# from data_handler.preprocess import get_news_feature, infer_news
# from data_handler.TestDataloader import DataLoaderTest
# from models.speedyrec import MLNR

# def run_evaluation(args):
#     setuplogger()
#     device = get_device()
    
#     if not args.load_ckpt_name:
#         raise ValueError("Thiếu checkpoint!")
        
#     logging.info(f"--> Loading checkpoint: {args.load_ckpt_name}")
#     checkpoint = torch.load(args.load_ckpt_name, map_location='cpu')
#     category_dict = checkpoint.get('category_dict', None)
#     subcategory_dict = checkpoint.get('subcategory_dict', None)
    
#     model = MLNR(args)
#     model.to(device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # --- DEBUG 1: KIỂM TRA NEWS FEATURE ---
#     logging.info("--> Encoding News...")
#     with torch.no_grad():
#         news_info, news_combined = get_news_feature(
#             args, mode='dev', category_dict=category_dict, subcategory_dict=subcategory_dict
#         )
#         news_vecs = infer_news(model, device, news_combined)
    
#     # IN LOG DEBUG QUAN TRỌNG
#     print(f"\n[DEBUG] Số lượng bài báo trong news_index: {len(news_info.news_index)}")
#     print(f"[DEBUG] Kích thước news_vecs: {news_vecs.shape}")
    
#     # Lấy thử 1 ID mẫu để xem format
#     sample_id = list(news_info.news_index.keys())[0]
#     print(f"[DEBUG] Sample News ID trong Dictionary: '{sample_id}'")
    
#     logging.info("--> Start Inference...")
#     dataloader = DataLoaderTest(
#         news_index=news_info.news_index,
#         news_scoring=news_vecs,
#         data_dirs=[os.path.join(args.root_data_dir, 'dev/')],
#         filename_pat=args.filename_pat, # Đảm bảo là "ProtoBuf_*.tsv"
#         args=args,
#         world_size=1,
#         worker_rank=0,
#         cuda_device_idx=0,
#         enable_prefetch=False, # TẮT PREFETCH ĐỂ DỄ DEBUG
#         enable_shuffle=False,
#         enable_gpu=args.enable_gpu,
#     )

#     results = MetricsDict(metrics_name=["AUC", "MRR", "nDCG5", "nDCG10"])
#     results.add_metric_dict('all users')
    
#     valid_batch_count = 0
#     total_samples = 0
#     skipped_samples = 0

#     with torch.no_grad():
#         for cnt, (log_vecs, log_mask, news_vecs_batch, labels) in enumerate(dataloader):
#             # --- DEBUG 2: KIỂM TRA BATCH ---
#             if cnt == 0:
#                 print(f"\n[DEBUG] Batch đầu tiên nhận được!")
#                 print(f"[DEBUG] Label shape: {len(labels)}")
#                 if len(labels) > 0:
#                     print(f"[DEBUG] Mẫu label đầu tiên: {labels[0]}")
#                     print(f"[DEBUG] Mean của label đầu tiên: {labels[0].mean()}")
            
#             if args.enable_gpu:
#                 log_vecs = log_vecs.cuda(device=device, non_blocking=True)
#                 log_mask = log_mask.cuda(device=device, non_blocking=True)

#             user_vecs = model.user_encoder(
#                 log_vecs, log_mask, user_log_mask=True
#             ).cpu().detach().numpy()

#             for i, (user_vec, news_vec, label) in enumerate(zip(user_vecs, news_vecs_batch, labels)):
                
#                 # --- DEBUG 3: TẠI SAO BỊ SKIP? ---
#                 # Check kỹ điều kiện skip
#                 if label.mean() == 0:
#                     if cnt == 0 and i < 5: print(f"[WARNING] Mẫu {i} bị Skip do TOÀN NEGATIVE (Mean=0)")
#                     skipped_samples += 1
#                     continue
#                 if label.mean() == 1:
#                     if cnt == 0 and i < 5: print(f"[WARNING] Mẫu {i} bị Skip do TOÀN POSITIVE (Mean=1)")
#                     skipped_samples += 1
#                     continue
                
#                 # Nếu qua được cửa ải này thì tính điểm
#                 total_samples += 1
#                 score = np.dot(news_vec, user_vec)
#                 metric_rslt = results.cal_metrics(score, label)
#                 results.update_metric_dict('all users', metric_rslt)
            
#             valid_batch_count += 1
#             if cnt % 100 == 0:
#                 print(f"Processed {cnt} batches...", end='\r')

#     print("\n" + "="*30)
#     print("CHẨN ĐOÁN CUỐI CÙNG")
#     print(f"Số batch chạy được: {valid_batch_count}")
#     print(f"Số mẫu tính điểm thành công: {total_samples}")
#     print(f"Số mẫu bị bỏ qua (Skip): {skipped_samples}")
#     print("="*30)

#     if total_samples > 0:
#         results.print_metrics(0, 0, 'all users')
#     else:
#         print("LỖI: KHÔNG CÓ MẪU NÀO HỢP LỆ ĐỂ TÍNH ĐIỂM!")
#         print("Gợi ý: Kiểm tra file ProtoBuf xem có cột Positive/Negative nào bị rỗng không.")

# if __name__ == '__main__':
#     args = parse_args()
#     args.world_size = 1
#     run_evaluation(args)
