import pandas as pd

files = [
    '/home/dungcony/projects/python/sumarization/data/original/train-00000-of-00001.parquet',
    '/home/dungcony/projects/python/sumarization/data/original/test-00000-of-00001.parquet',
    '/home/dungcony/projects/python/sumarization/data/original/valid-00000-of-00001.parquet'
]

def analyze_parquet():
    for file in files:
        print(f"=== Analyzing {file.split('/')[-1]} ===")
        try:
            df = pd.read_parquet(file)
            
            print(f"Columns: {list(df.columns)}")
            print(f"Total Rows: {len(df)}")
            
            # Assume columns are similar, e.g. Document, Summary
            doc_col = None
            for col in ['Document', 'article', 'text', 'source']:
                if col in df.columns:
                    doc_col = col
                    break
                    
            sum_col = None
            for col in ['Summary', 'summary', 'target']:
                if col in df.columns:
                    sum_col = col
                    break
            
            if doc_col and sum_col:
                # Check empty strings
                empty_docs = (df[doc_col].astype(str).str.strip() == '').sum()
                empty_sums = (df[sum_col].astype(str).str.strip() == '').sum()
                print(f"Empty Docs: {empty_docs}, Empty Sums: {empty_sums}")
                
                # Duplicates
                exact_dups = df.duplicated().sum()
                doc_dups = df.duplicated(subset=[doc_col]).sum()
                print(f"Exact Duplicate Rows: {exact_dups}, Duplicate Docs: {doc_dups - exact_dups}")
                
                # Length check
                d_len = df[doc_col].astype(str).str.split().str.len()
                s_len = df[sum_col].astype(str).str.split().str.len()
                
                print(f"Doc length - Min: {d_len.min()}, Max: {d_len.max()}, Mean: {d_len.mean():.1f}")
                print(f"Sum length - Min: {s_len.min()}, Max: {s_len.max()}, Mean: {s_len.mean():.1f}")
                print(f"Summaries longer than docs: {(s_len > d_len).sum()}")
                print(f"Docs < 10 words: {(d_len < 10).sum()}")
                
                # HTML tags and LaTeX
                html_tags = df[doc_col].astype(str).str.contains(r'<.*?>', regex=True).sum()
                latex_tags = df[doc_col].astype(str).str.contains(r'{\\', regex=False).sum()
                print(f"Docs with HTML tags: {html_tags}")
                print(f"Docs with LaTeX tags (e.g. {{\\): {latex_tags}")
                
                # 'loại' at start
                loai_start = df[doc_col].astype(str).str.contains(r'(?i)^loại[\s\n\r]+', regex=True).sum()
                print(f"Docs starting with 'loại': {loai_start}")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
        print("\n")

if __name__ == "__main__":
    analyze_parquet()
