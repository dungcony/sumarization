import os
import csv
import glob
import re
import random

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
# ==========================================
DATA_DIR = '/home/dungcony/projects/python/sumarization/data/fine-turn'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw') # Nơi chứa dữ liệu thô ban đầu
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
VAL_DIR = os.path.join(DATA_DIR, 'validation')

# ==========================================
# CÁC HÀM XỬ LÝ LÀM SẠCH (CLEANING FUNCTIONS)
# ==========================================
def word_count(s):
    """Đếm số từ trong đoạn văn bản."""
    return len(s.split())

def remove_html_tags(text):
    """Xoá các thẻ HTML như <a>, <i>, <br>..."""
    clean = re.compile('<.*?>')
    return re.sub(clean, ' ', text)

def remove_latex_blocks(text):
    """Xoá các khối mã công thức Toán học (LaTeX/MediaWiki) bắt đầu bằng '{\\'"""
    while '{\\' in text:
        start = text.find('{\\')
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            text = text[:start] + ' ' + text[end+1:]
        else:
            text = text.replace('{\\', '', 1)
    return text

def clean_text(text):
    """Quy trình làm sạch tổng hợp cho một đoạn văn bản."""
    # Xoá HTML & LaTeX
    text = remove_html_tags(text)
    text = remove_latex_blocks(text)
    
    # Xoá từ "loại" hoặc "Loại" ở ngay đầu câu (Do lỗi crawl từ Wiki)
    text = re.sub(r'^loại[\s\n\r]+', '', text, flags=re.IGNORECASE)
    
    # Xoá các khoảng trắng, dấu enter, tab thừa và gom lại thành 1 dấu cách
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# QUY TRÌNH XỬ LÝ CHÍNH
# ==========================================
def process_and_split():
    # Khởi tạo các thư mục đầu ra
    for d in [TRAIN_DIR, TEST_DIR, VAL_DIR]:
        os.makedirs(d, exist_ok=True)
        
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))
    if not csv_files:
        print(f"Không tìm thấy file .csv nào trong thư mục: {RAW_DATA_DIR}")
        print("Vui lòng đặt các file data gốc vào thư mục 'raw' để chạy script này.")
        return
        
    for file in csv_files:
        filename = os.path.basename(file)
        print(f"\nĐang xử lý file: {filename}")
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                if 'Document' not in header or 'Summary' not in header:
                    print(f"  -> Bỏ qua vì thiếu cột Document hoặc Summary")
                    continue
                    
                # Lấy index của các cột quan trọng
                doc_idx = header.index('Document')
                sum_idx = header.index('Summary')
                
                dataset_idx = -1
                if 'Dataset' in header:
                    dataset_idx = header.index('Dataset')
                elif 'dataset' in header:
                    dataset_idx = header.index('dataset')
                    
                # Cập nhật Header mới
                new_header = []
                for i, col_name in enumerate(header):
                    if i == dataset_idx:
                        continue # Bỏ cột Dataset
                    elif i == doc_idx:
                        new_header.append('article') # Đổi Document -> article
                    else:
                        new_header.append(col_name)
                        
                seen_articles = set()
                cleaned_rows = []
                original_count = 0
                
                # Duyệt qua từng dòng dữ liệu
                for row in reader:
                    # Bỏ qua nếu dòng bị thiếu cột so với header
                    if len(row) <= max(doc_idx, sum_idx):
                        continue
                    original_count += 1
                    
                    doc = row[doc_idx]
                    summ = row[sum_idx]
                    
                    # BƯỚC 1: Bỏ các dòng rỗng
                    if not doc.strip() or not summ.strip():
                        continue
                        
                    # BƯỚC 2: Làm sạch Text
                    doc = clean_text(doc)
                    summ = clean_text(summ)
                    
                    # BƯỚC 3: Lọc bài viết quá ngắn hoặc bản tóm tắt dài hơn bài gốc
                    d_len = word_count(doc)
                    s_len = word_count(summ)
                    if d_len < 10 or s_len > d_len:
                        continue
                        
                    # BƯỚC 4: Bỏ các dòng trùng lặp (dựa trên nội dung article)
                    if doc in seen_articles:
                        continue
                    seen_articles.add(doc)
                    
                    # BƯỚC 5: Tái tạo lại dòng dữ liệu mới theo cấu trúc new_header
                    new_row = []
                    for i in range(len(row)):
                        if i == dataset_idx:
                            continue
                        elif i == doc_idx:
                            new_row.append(doc)
                        elif i == sum_idx:
                            new_row.append(summ)
                        else:
                            new_row.append(row[i])
                            
                    cleaned_rows.append(new_row)
                    
            # BƯỚC 6: Trộn ngẫu nhiên (Shuffle)
            random.seed(42)
            random.shuffle(cleaned_rows)
            
            # BƯỚC 7: Chia tách Train(80%) - Test(10%) - Validation(10%)
            train_end = int(len(cleaned_rows) * 0.8)
            test_end = int(len(cleaned_rows) * 0.9)
            
            train_rows = cleaned_rows[:train_end]
            test_rows = cleaned_rows[train_end:test_end]
            val_rows = cleaned_rows[test_end:]
            
            # BƯỚC 8: Lưu ra file
            def write_csv(path, header_cols, data_rows):
                with open(path, 'w', encoding='utf-8', newline='') as out_f:
                    writer = csv.writer(out_f)
                    writer.writerow(header_cols)
                    writer.writerows(data_rows)
                    
            write_csv(os.path.join(TRAIN_DIR, filename), new_header, train_rows)
            write_csv(os.path.join(TEST_DIR, filename), new_header, test_rows)
            write_csv(os.path.join(VAL_DIR, filename), new_header, val_rows)
            
            print(f"  -> Đã làm sạch: {original_count} dòng gốc -> {len(cleaned_rows)} dòng sạch.")
            print(f"  -> Đã lưu Train: {len(train_rows)} | Test: {len(test_rows)} | Val: {len(val_rows)}")
            
        except Exception as e:
            print(f"  -> Lỗi khi xử lý file {filename}: {e}")

if __name__ == "__main__":
    print("BẮT ĐẦU QUY TRÌNH LÀM SẠCH VÀ CHIA TÁCH DỮ LIỆU...")
    process_and_split()
    print("\nHOÀN THÀNH!")
