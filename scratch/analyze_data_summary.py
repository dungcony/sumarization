import pandas as pd

def analyze():
    print("Loading data_summary.csv ...")
    try:
        # Load the whole dataset or a chunk
        df = pd.read_csv('/home/dungcony/projects/python/sumarization/data/original/data_summary.csv', on_bad_lines='skip')
        print(f"Total Rows parsed: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        doc_col = 'Text' if 'Text' in df.columns else 'article'
        sum_col = 'Summary' if 'Summary' in df.columns else 'summary'
        
        if doc_col not in df.columns or sum_col not in df.columns:
            print("Columns not found!")
            return
            
        df[doc_col] = df[doc_col].astype(str)
        df[sum_col] = df[sum_col].astype(str)
        
        # Checking Empty
        empty_docs = (df[doc_col].str.strip() == '').sum()
        empty_sums = (df[sum_col].str.strip() == '').sum()
        print(f"Empty Docs: {empty_docs}, Empty Sums: {empty_sums}")
        
        # Checking HTML
        html_tags = df[doc_col].str.contains(r'<.*?>', regex=True).sum()
        print(f"Docs with HTML tags: {html_tags}")
        
        # Checking LaTeX
        latex_tags = df[doc_col].str.contains(r'{\\', regex=False).sum()
        print(f"Docs with LaTeX tags (e.g. {{\\): {latex_tags}")
        
        # Lengths
        d_len = df[doc_col].str.split().str.len()
        s_len = df[sum_col].str.split().str.len()
        
        print(f"Doc length - Min: {d_len.min()}, Max: {d_len.max()}, Mean: {d_len.mean():.1f}")
        print(f"Sum length - Min: {s_len.min()}, Max: {s_len.max()}, Mean: {s_len.mean():.1f}")
        print(f"Docs < 10 words: {(d_len < 10).sum()}")
        print(f"Summaries > Docs: {(s_len > d_len).sum()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze()
