import json
import os
import argparse
import multiprocessing as mp
import torch
import numpy as np
from tqdm import tqdm
import logging
import sys
import copy
import traceback

# ================= 1. Environment Setup =================
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from verl.utils import hf_processor
    from qwen_omni_utils import process_audio_info
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Error: Cannot import environment components. Please run in the correct environment.")

# ================= 2. Helper Functions =================
def get_media_info(history):
    """Extract the first media path from history for error location"""
    for turn in history:
        for item in turn.get('content', []):
            if isinstance(item, dict):
                for k in ['video', 'image', 'audio']:
                    if k in item: return item[k]
    return "No_Media_Found"

# ================= 3. Worker Core Logic =================
_PROCESSOR = None

def init_worker(model_path):
    """Worker initialization: fully silence output and load processor"""
    global _PROCESSOR
    fnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(fnull, sys.stdout.fileno())
    os.dup2(fnull, sys.stderr.fileno())
    logging.disable(logging.CRITICAL)
    _PROCESSOR = hf_processor(model_path, trust_remote_code=False, use_fast=True)
    torch.set_grad_enabled(False)

def process_line(indexed_line):
    """Rigorous computation logic, returns status, details, and result"""
    idx, line = indexed_line
    global _PROCESSOR
    if _PROCESSOR is None: return "Processor_Init_Error", {"idx": idx}, None
    
    media_path = "Unknown"
    try:
        data = json.loads(line.strip())
        history = copy.deepcopy(data.get('raw_input', []))
        media_path = get_media_info(history)
        output_text = data.get('output', "")
        has_audio = data.get('extra_info', {}).get('has_audio', False)

        # A. Prompt computation (with heavy media processing)
        prompt_text = _PROCESSOR.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        if isinstance(prompt_text, list): prompt_text = "".join(prompt_text)
        
        img_in, vid_in = process_vision_info(history)
        aud_in = process_audio_info(history, use_audio_in_video=True) if has_audio else None
        
        kw_p = {"text": [prompt_text], "images": img_in, "videos": vid_in, "audio": aud_in, "use_audio_in_video": bool(aud_in), "return_tensors": "pt"}
        
        with torch.no_grad():
            out_p = _PROCESSOR(**kw_p)
            prompt_tokens = int(out_p["input_ids"][0].numel())

        # B. Full Sequence computation (lightweight text delta)
        full_history = history + [{"role": "assistant", "content": [{"type": "text", "text": str(output_text)}]}]
        full_text = _PROCESSOR.apply_chat_template(full_history, add_generation_prompt=False, tokenize=False)
        if isinstance(full_text, list): full_text = "".join(full_text)
        
        p_ids = _PROCESSOR.tokenizer.encode(str(prompt_text), add_special_tokens=False)
        f_ids = _PROCESSOR.tokenizer.encode(str(full_text), add_special_tokens=False)
        
        delta_tokens = len(f_ids) - len(p_ids)
        total_tokens = prompt_tokens + delta_tokens

        # C. Package result
        data['extra_info']['token_stats']['final_history_tokens'] = prompt_tokens
        data['extra_info']['token_stats']['full_sequence_tokens'] = total_tokens
        
        return "SUCCESS", {"idx": idx}, (json.dumps(data, ensure_ascii=False), prompt_tokens, total_tokens)
    
    except Exception as e:
        error_info = {
            "idx": idx,
            "path": media_path,
            "err_type": type(e).__name__,
            "err_msg": str(e).split('\n')[0],
            "trace": traceback.format_exc().split('\n')[-3:-1]
        }
        return "ERROR", error_info, None

# ================= 4. Statistics Report =================
def print_report(p_list, t_list, file_name, total_req, fail_details, is_subset=False):
    if not t_list and not fail_details: return
    
    success_count = len(t_list)
    fail_count = total_req - success_count
    over_32k = sum(1 for x in t_list if x > 32768) if t_list else 0
    
    print(f"\n" + "="*95)
    print(f"Token Deep Statistics Report {'(Top Subset Mode)' if is_subset else '(Full Mode)'}")
    print(f"File: {file_name}")
    print(f"="*95)
    
    if t_list:
        print(f"{'Metric':<20} | {'Prompt (Input)':<25} | {'Total (Full Seq)':<25}")
        print(f"-"*85)
        for name, func in [("Min", np.min), ("Max", np.max), ("Mean", np.mean), ("Median", np.median)]:
            print(f"{name:<20} | {int(func(p_list)):<25} | {int(func(t_list)):<25}")
        print(f"-"*85)

    print(f"{'Total Requested':<20} | {total_req:<25}")
    print(f"{'Success Processed':<20} | {success_count:<25} ({(success_count/total_req)*100:.2f}%)")
    print(f"{'Failed/Skipped':<20} | {fail_count:<25} ({(fail_count/total_req)*100:.2f}%)")
    if success_count > 0:
        print(f"{'Over 32K (32768)':<20} | {over_32k:<25} ({(over_32k/success_count)*100:.2f}%)")
    
    if fail_details:
        print(f"-"*85)
        print(f"Failure Reason Distribution (Top 5 Reasons):")
        reasons = {}
        for d in fail_details: reasons[d['err_msg']] = reasons.get(d['err_msg'], 0) + 1
        for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {r[:60]:<60}: {c} items")
        
        print(f"\nFailed Sample Details (First 3):")
        for d in fail_details[:3]:
            print(f"   - [Line:{d['idx']}] Path: {d['path']}")
            print(f"     Error: {d['err_type']}: {d['err_msg']}")

    if t_list:
        print(f"-"*85)
        print(f"[Recomputed Total Token Peak Ranking]")
        t_sorted = sorted(t_list, reverse=True)
        for i in range(min(5, len(t_sorted))):
            print(f" Rank {i+1:<15} | {t_sorted[i]:<25}")
    
    print(f"="*95 + "\n")

# ================= 5. Main Program =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--top", type=int, default=0)
    args = parser.parse_args()

    debug_file = args.output.replace(".jsonl", "_DEBUG_DETAILS.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"Scanning raw data and building index...")
    all_lines_indexed = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                temp_data = json.loads(line)
                old_val = temp_data.get('extra_info', {}).get('token_stats', {}).get('final_history_tokens', 0)
                all_lines_indexed.append((old_val, i, line))
            except: pass

    all_lines_indexed.sort(key=lambda x: x[0], reverse=True)
    target_data = all_lines_indexed[:args.top] if args.top > 0 else all_lines_indexed
    tasks = [(x[1], x[2]) for x in target_data]

    p_list, t_list, fail_details = [], [], []

    print(f"Processing samples (details in {os.path.basename(debug_file)})...")
    with open(args.output, 'w', encoding='utf-8') as f_out, open(debug_file, 'w', encoding='utf-8') as f_err:
        with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(args.model_path,)) as pool:
            iterator = pool.imap(process_line, tasks, chunksize=args.chunk)
            for status, info, res in tqdm(iterator, total=len(tasks), desc="Processing"):
                if status == "SUCCESS":
                    res_json, p_tok, t_tok = res
                    f_out.write(res_json + "\n")
                    p_list.append(p_tok)
                    t_list.append(t_tok)
                else:
                    f_err.write(json.dumps(info, ensure_ascii=False) + "\n")
                    fail_details.append(info)

    print_report(p_list, t_list, os.path.basename(args.output), len(tasks), fail_details, is_subset=(args.top > 0))

if __name__ == "__main__":
    main()


'''
Example:
python recompute_and_stat_tokens.py \
    --input "train_steps_final_sft.jsonl" \
    --output "../results_token/steps_recomputed.jsonl" \
    --model_path "/path/to/Qwen2.5-Omni-7B" \
    --workers 32 \
    --top 320 \
    --chunk 1
'''
