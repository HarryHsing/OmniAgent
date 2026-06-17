import json
import random
import os
import argparse
from collections import Counter

def load_jsonl(path, filter_key=None, max_duration=None):
    if not path or not os.path.exists(path):
        if path:
            print(f"Warning: File not found {path}")
        return []
    
    data = []
    filtered_duration = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 1. Question type filtering
                if filter_key:
                    if filter_key not in item.get("question_type", ""):
                        continue
                
                # 2. Duration filtering
                if max_duration is not None:
                    # Compatible with both duration and video_duration field names
                    duration = item.get("duration_seconds", item.get("video_duration", None))
                    if duration is not None and float(duration) > max_duration:
                        filtered_duration += 1
                        # print(f"Filtered duration {duration}s > {max_duration}s sample, video_path: {item.get('video', 'N/A')}")
                        continue
                
                data.append(item)
            except:
                continue
                
    random.shuffle(data)
    msg = f"Loaded {len(data)} items: {os.path.basename(path)}"
    if filtered_duration > 0:
        msg += f" (filtered duration > {max_duration}s: {filtered_duration} items)"
    print(msg)
    return data

def pop_samples(bucket, n):
    """Pop n samples from the bucket"""
    count = min(len(bucket), n)
    samples = bucket[:count]
    del bucket[:count]
    return samples

def main():
    parser = argparse.ArgumentParser(description="Strict ratio dataset merging (with duration filtering)")
    
    # Ratio parameters
    parser.add_argument("--mcq_count", type=int, default=20, help="MCQ count per batch")
    parser.add_argument("--tr_count", type=int, default=8, help="TR count per batch")
    parser.add_argument("--size_count", type=int, default=4, help="SIZE count per batch")
    parser.add_argument("--max_duration", type=float, default=300.0, help="Max video duration (seconds)")
    parser.add_argument("--output", type=str, default="merged_rl_final.jsonl")
    
    # File paths
    parser.add_argument("--holmes", type=str)
    parser.add_argument("--vsi", type=str)
    parser.add_argument("--vr_l", type=str)
    parser.add_argument("--vr_s", type=str)
    parser.add_argument("--multihop", type=str)
    parser.add_argument("--vale_l", type=str)
    parser.add_argument("--vale_s", type=str)
    parser.add_argument("--vsi_size", type=str)

    args = parser.parse_args()

    # 1. Load all buckets (with duration filtering)
    buckets = {
        "Holmes":   load_jsonl(args.holmes, max_duration=args.max_duration),
        "VSI_MCQ":  load_jsonl(args.vsi, filter_key="MCQ_VSI", max_duration=args.max_duration),
        "VR_L":     load_jsonl(args.vr_l, max_duration=args.max_duration),
        "VR_S":     load_jsonl(args.vr_s, max_duration=args.max_duration),
        "MultiHop": load_jsonl(args.multihop, max_duration=args.max_duration),
        "VALE_L":   load_jsonl(args.vale_l, max_duration=args.max_duration),
        "VALE_S":   load_jsonl(args.vale_s, max_duration=args.max_duration),
        "VSI_SIZE": load_jsonl(args.vsi_size, filter_key="SIZE_VSI", max_duration=args.max_duration),
    }

    final_data = []
    batch_count = 0
    usage_stats = Counter()
    
    print(f"\nTarget ratio -> MCQ:{args.mcq_count} | TR:{args.tr_count} | SIZE:{args.size_count}")
    print(f"Duration limit -> <= {args.max_duration}s")

    # 2. Generate batches in loop
    while True:
        # Compute remaining totals for three categories (fixed VSI_SIZE reference)
        rem_mcq  = len(buckets["Holmes"]) + len(buckets["VSI_MCQ"]) + len(buckets["VR_L"]) + len(buckets["VR_S"])
        rem_tr   = len(buckets["MultiHop"]) + len(buckets["VALE_L"]) + len(buckets["VALE_S"])
        rem_size = len(buckets["VSI_SIZE"])

        # Core logic: stop immediately if any category is insufficient
        if rem_mcq < args.mcq_count or rem_tr < args.tr_count or rem_size < args.size_count:
            print(f"\nStopping: Insufficient data for a complete batch. Remaining -> MCQ:{rem_mcq}, TR:{rem_tr}, SIZE:{rem_size}")
            break

        batch = []

        # --- MCQ portion (prioritize Holmes and VSI, VR as fallback) ---
        m_samples = []
        m_samples.extend(pop_samples(buckets["Holmes"], 4))
        m_samples.extend(pop_samples(buckets["VSI_MCQ"], 4))
        gap = args.mcq_count - len(m_samples)
        if gap > 0:
            gl, gs = gap // 2, gap - (gap // 2)
            l_s = pop_samples(buckets["VR_L"], gl)
            s_s = pop_samples(buckets["VR_S"], gs)
            # Internal backfill
            if len(l_s) < gl: s_s.extend(pop_samples(buckets["VR_S"], gl - len(l_s)))
            elif len(s_s) < gs: l_s.extend(pop_samples(buckets["VR_L"], gs - len(s_s)))
            m_samples.extend(l_s + s_s)
        batch.extend(m_samples)
        usage_stats["MCQ"] += len(m_samples)

        # --- TR portion (prioritize MultiHop, VALE as fallback) ---
        t_samples = pop_samples(buckets["MultiHop"], 2)
        gap = args.tr_count - len(t_samples)
        if gap > 0:
            gl, gs = gap // 2, gap - (gap // 2)
            tl = pop_samples(buckets["VALE_L"], gl)
            ts = pop_samples(buckets["VALE_S"], gs)
            if len(tl) < gl: ts.extend(pop_samples(buckets["VALE_S"], gl - len(tl)))
            elif len(ts) < gs: tl.extend(pop_samples(buckets["VALE_L"], gs - len(ts)))
            t_samples.extend(tl + ts)
        batch.extend(t_samples)
        usage_stats["TR"] += len(t_samples)

        # --- SIZE portion ---
        sz_samples = pop_samples(buckets["VSI_SIZE"], args.size_count)
        batch.extend(sz_samples)
        usage_stats["SIZE"] += len(sz_samples)

        # --- Shuffle within each chunk ---
        random.shuffle(batch)
        final_data.extend(batch)
        batch_count += 1

    # 3. Write to file
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "="*50)
    print(f"Merge successful!")
    print(f"Total batches: {batch_count}")
    print(f"Total samples: {len(final_data)}")
    print(f"Final ratio:   MCQ:{args.mcq_count} : TR:{args.tr_count} : SIZE:{args.size_count}")
    print("="*50)

if __name__ == "__main__":
    main()


'''
    
python merge_tasks_for_rl_train.py \
    --holmes train_video-holmes_converted_gemini-3-pro-preview_high_s32_f60_a300.0_c60.0_OmniAgent-RAND-DYN-RC_p90_v3_steps_rl_negatives.jsonl \
    --vsi train_vsi_train_10k_clean_rl_balanced_converted_gemini-3-pro-preview_high_s32_f60_a300.0_c60.0_OmniAgent-RAND-DYN-RC_p90_v3_steps_rl_negatives.jsonl \
    --vr_s train_LongVideoReason_0s_300s_omniagent_gemini-3-pro-preview_high_s32_f60_a300.0_c60.0_OmniAgent-RAND-DYN-RC_p90_v3_steps_rl_negatives.jsonl \
    --multihop train_MultiHop-EgoQA_omniagent_gemini-3-pro-preview_high_s32_f60_a300.0_c60.0_OmniAgent-RAND-DYN-RC_p90_v3_steps_rl_negatives.jsonl \
    --vale_s train_LongVALE_0s_300s_omniagent_gemini-3-pro-preview_high_s32_f60_a300.0_c60.0_OmniAgent-RAND-DYN-RC_p90_v3_steps_rl_negatives.jsonl \
    --vsi_size train_vsi_train_10k_clean_rl_balanced_converted_gemini-3-pro-preview_high_s32_f60_a300.0_c60.0_OmniAgent-RAND-DYN-RC_p90_v3_steps_rl_negatives.jsonl \
    --mcq_count 24 \
    --tr_count 6 \
    --size_count 2 \
    --max_duration 300 \
    --output results_v1-2_rl_balanced.jsonl
'''