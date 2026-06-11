import json
import os
import numpy as np
import argparse

def get_token_stats(file_path):
    """
    Analyze token distribution across all turns in a single JSONL file and extract top three values.
    """
    tokens = []
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Extract extra_info -> token_stats -> final_history_tokens
                e_info = data.get('extra_info', {})
                t_stats = e_info.get('token_stats', {})
                
                t = t_stats.get('final_history_tokens') if isinstance(t_stats, dict) else e_info.get('final_history_tokens')
                
                if t is not None:
                    tokens.append(int(t))
            except Exception:
                continue
    
    if not tokens:
        return None
    
    # Sort to get Top N
    sorted_tokens = sorted(tokens, reverse=True)
    top_1 = sorted_tokens[0]
    top_2 = sorted_tokens[1] if len(sorted_tokens) > 1 else "-"
    top_3 = sorted_tokens[2] if len(sorted_tokens) > 2 else "-"
        
    return {
        "min": int(min(tokens)),
        "avg": int(sum(tokens) / len(tokens)),
        "median": int(np.median(tokens)),
        "count": len(tokens),
        "top1": top_1,
        "top2": top_2,
        "top3": top_3
    }

def main():
    parser = argparse.ArgumentParser(description="Standalone token consumption statistics and top peak values for JSONL files")
    parser.add_argument("--dir", type=str, required=True, help="Directory path containing jsonl files")
    parser.add_argument("--file", type=str, default=None, help="Optional: only analyze a specific jsonl file")
    args = parser.parse_args()

    target_dir = args.dir
    
    # Adjust table width to accommodate Top 1, 2, 3
    header_str = f"{'File Name':<45} | {'Turns':<6} | {'Avg':<7} | {'Med':<7} | {'Top 1':<8} | {'Top 2':<8} | {'Top 3':<8}"
    divider = "-" * len(header_str)

    print(f"\n" + "="*len(header_str))
    print(f"Token Consumption Depth Report (Field: final_history_tokens)")
    print(f"Target Directory: {target_dir}")
    print("="*len(header_str))
    print(header_str)
    print(divider)

    # Determine file list to process
    if args.file:
        files = [args.file] if os.path.isabs(args.file) else [os.path.join(target_dir, args.file)]
    else:
        files = [os.path.join(target_dir, f) for f in sorted(os.listdir(target_dir)) if f.endswith(".jsonl")]

    found_any = False
    for f_path in files:
        f_name = os.path.basename(f_path)
        res = get_token_stats(f_path)
        if res:
            found_any = True
            # Format Top 2, 3 output (handle cases with fewer than 3 entries)
            t1, t2, t3 = res['top1'], res['top2'], res['top3']
            
            print(f"{f_name[:44]:<45} | {res['count']:<6} | {res['avg']:<7} | {res['median']:<7} | {t1:<8} | {t2:<8} | {t3:<8}")
    
    if not found_any:
        print("No .jsonl files with valid token data found in the specified directory.")
    
    print(divider + "\n")

if __name__ == "__main__":
    main()


'''
python stat_tokens.py --dir "/path/to/results_final_v1"

'''
