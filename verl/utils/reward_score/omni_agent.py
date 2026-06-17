# omniagent reward
import os, time, json, requests, ast
from dotenv import load_dotenv

load_dotenv()                                       # Load variables from .env
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
API_KEY  = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OSS_ACCESS_KEY")
if not API_KEY:
    import warnings
    warnings.warn(
        "DASHSCOPE_API_KEY not set. Free-form (FF) scoring requires an LLM judge API. "
        "FF questions will receive reward=0. Set DASHSCOPE_API_KEY in your .env file "
        "or environment to enable LLM-as-judge scoring.",
        stacklevel=2,
    )

MODEL_NAME = "gpt-5-2025-08-07" #"gpt-4.1-2025-04-14"

def gpt_api(prompt: str,
            model_name: str = None,
            stream: bool = False,
            max_try: int = 100,
            timeout: int = 15) -> str:
    """Call DashScope compatible-mode chat completion API."""
    if not API_KEY:
        raise RuntimeError("gpt_api called but no API key is configured")

    # Build messages (compatible-mode requires content as a list)
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }]

    payload = {
        "model": model_name,
        "stream": stream,
        "messages": messages,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY,                 # No Bearer prefix needed
        "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}',
    }

    tries = 0
    while tries < max_try:
        try:
            r = requests.post(BASE_URL,
                              headers=headers,
                              data=json.dumps(payload),
                              timeout=timeout,
                              stream=stream)

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")

            # Parse response
            if not stream:
                return r.json()["choices"][0]["message"]["content"]

            # Streaming: read line by line
            full = []
            for ln in r.iter_lines(decode_unicode=True):
                if not ln or ln.strip() == "data: [DONE]":
                    continue
                if ln.startswith("data: "):
                    delta = json.loads(ln[6:])["choices"][0]["delta"]
                    if "content" in delta:
                        full.append(delta["content"])
            return "".join(full)

        except Exception as e:
            print(f"[WARN] gpt_api retry {tries+1}/{max_try}: {e}")
            tries += 1
            time.sleep(1)

    raise RuntimeError("gpt_api failed after all retries")


def compute_score_free_form(output_ans: str, gt_ans: str, question_text: str) -> float:
    if not API_KEY:
        print("[WARNING] No API key configured — FF reward defaults to 0.0")
        return 0.0

    full_prompt = get_prompt(output_ans, gt_ans, question_text)

    response = gpt_api(prompt=full_prompt, model_name=MODEL_NAME)

    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(output_ans) >= 1000:
        acc_reward = 0.0

    return acc_reward


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


#########################################################################################
# omni_agent.py

def evaluate_reasoning_quality(history, question, answer, ground_truth, question_type, options=None, max_retry=100):
    """
    Evaluate reasoning quality (strict check).

    Args:
        history: Full conversation history (may contain media)
        question: Question text
        answer: Model's answer
        ground_truth: Ground-truth answer
        question_type: Question type
        options: Option list (used for MCQ)
        max_retry: Maximum retry count

    Returns:
        {
            "pass_quality_check": bool,  # True/False
            "reason": str                # Reason for failure
        }
    """
    if not API_KEY:
        return {"pass_quality_check": True, "reason": "skipped: no API key"}

    # Clean all media from history
    cleaned_history = clean_history_media(history)

    # Build options text (for MCQ)
    options_text = ""
    if question_type == "MCQ":
        options_text = "\n**Options**:\n" + "\n".join(options)

    prompt = f"""You are a strict reasoning quality evaluator. Evaluate the reasoning process and answer quality.

**Question**: {question}{options_text}
**Question Type**: {question_type}
**Ground Truth**: {ground_truth}
**Student Answer**: {answer}

**Reasoning History** (text only, media omitted):
{format_history_for_prompt(cleaned_history)}

**Evaluation Criteria (STRICT - any ONE failure → reject):**

1. **Logical Reasoning**: Steps must be coherent, conclusions must follow from evidence
2. **Evidence Quality**: Must gather relevant evidence before answering
3. **No Hallucination**: All claims must be supported by gathered evidence
4. **No Garbage Characters**: Check for �, \\x00, unprintable chars, or encoding errors
5. **No Mixed Languages**: Should use consistent language (no unexpected Chinese/Japanese/etc in English context)

**Output Format (JSON only):**
{{
    "pass_quality_check": true/false,
    "reason": "specific reason if failed, empty if passed"
}}

Only output the JSON, nothing else.
"""

    for attempt in range(max_retry):
        response = None
        try:
            response = gpt_api(prompt=prompt, model_name=MODEL_NAME, timeout=30)

            result = json.loads(response.strip())

            if "pass_quality_check" not in result:
                print(f"[WARN] evaluate_reasoning_quality attempt {attempt+1}/{max_retry}: "
                      f"missing 'pass_quality_check' in response: {result}")
                if attempt < max_retry - 1:
                    time.sleep(1)
                    continue
                else:
                    return {
                        "pass_quality_check": False,
                        "reason": "API response format error after retries"
                    }

            return result

        except json.JSONDecodeError as e:
            print(f"[WARN] evaluate_reasoning_quality attempt {attempt+1}/{max_retry}: "
                  f"JSON parse failed: {e}, response={response}")
            if attempt < max_retry - 1:
                time.sleep(1)
                continue
            else:
                return {
                    "pass_quality_check": False,
                    "reason": f"API response parse error after {max_retry} retries"
                }

        except Exception as e:
            print(f"[WARN] evaluate_reasoning_quality attempt {attempt+1}/{max_retry}: "
                  f"API call failed: {e}")
            if attempt < max_retry - 1:
                time.sleep(1)
                continue
            else:
                return {
                    "pass_quality_check": False,
                    "reason": f"API call failed after {max_retry} retries: {str(e)}"
                }

    return {
        "pass_quality_check": False,
        "reason": "Unknown error in evaluation"
    }


def clean_history_media(history):
    """Remove all media (image/video/audio) from history, keeping only text."""
    cleaned = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", [])

        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        text_only = [part for part in content if part.get("type") == "text"]

        if text_only:
            cleaned.append({"role": role, "content": text_only})

    return cleaned


def format_history_for_prompt(history):
    """Format history as human-readable text."""
    lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", [])

        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if part.get("type") == "text"]
            text = " ".join(texts)
        else:
            text = str(content)

        lines.append(f"[{role.upper()}]: {text}")

    return "\n".join(lines)

###################################################
if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: compute_score_free_form (answer correctness)")
    print("=" * 60)
    score = compute_score_free_form('light blue', "Blue", "which color is of sky?")
    print(f"Answer: 'light blue', GT: 'Blue', Score: {score}\n")

    print("=" * 60)
    print("Test 2: evaluate_reasoning_quality (good reasoning)")
    print("=" * 60)

    good_history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful video QA assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Video META:\n- duration: 30s\n- fps: 25\n\nQuestion: What color is the car in the video?"}]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": '{"observation": "", "think": "I need to examine the video to see the car", "action": {"type": "get_frames", "timestamps": [5.0, 10.0, 15.0]}}'
            }]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "http://example.com/frame1.jpg"},
                {"type": "text", "text": "frames 5.00s/10.00s/15.00s"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": '{"observation": "I can see a red car in all three frames", "think": "The car is consistently red across different timestamps", "action": {"type": "answer", "content": "red"}}'
            }]
        },
    ]

    result_good = evaluate_reasoning_quality(
        history=good_history,
        question="What color is the car in the video?",
        answer="red",
        ground_truth="red",
        question_type="FF"
    )
    print(f"Good reasoning result: {json.dumps(result_good, indent=2, ensure_ascii=False)}\n")


    print("=" * 60)
    print("Test 3: evaluate_reasoning_quality (garbage characters)")
    print("=" * 60)

    garbage_history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful video QA assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Question: What is the person doing?"}]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": '{"observation": "", "think": "I need�to check", "action": {"type": "get_frames", "timestamps": [5.0]}}'
            }]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "http://example.com/frame.jpg"},
                {"type": "text", "text": "frames 5.00s"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": '{"observation": "Person is running", "think": "Clear�action", "action": {"type": "answer", "content": "running"}}'
            }]
        },
    ]

    result_garbage = evaluate_reasoning_quality(
        history=garbage_history,
        question="What is the person doing?",
        answer="running",
        ground_truth="running",
        question_type="FF"
    )
    print(f"Garbage reasoning result: {json.dumps(result_garbage, indent=2, ensure_ascii=False)}\n")


    print("=" * 60)
    print("Test 4: evaluate_reasoning_quality (no logic)")
    print("=" * 60)

    illogical_history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful video QA assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Video META:\n- duration: 30s\n\nQuestion: How many people are in the video?"}]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": '{"observation": "", "think": "I guess there are 3 people", "action": {"type": "answer", "content": "3"}}'
            }]
        },
    ]

    result_illogical = evaluate_reasoning_quality(
        history=illogical_history,
        question="How many people are in the video?",
        answer="3",
        ground_truth="3",
        question_type="FF"
    )
    print(f"Illogical reasoning result: {json.dumps(result_illogical, indent=2, ensure_ascii=False)}\n")


    print("=" * 60)
    print("Test 5: evaluate_reasoning_quality (mixed languages)")
    print("=" * 60)

    mixed_lang_history = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful video QA assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Question: What is the weather like?"}]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": '{"observation": "", "think": "我需要检查天气情况", "action": {"type": "get_frames", "timestamps": [5.0]}}'
            }]
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "http://example.com/clip.mp4"},
                {"type": "text", "text": "clip 0-10"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": '{"observation": "It is sunny", "think": "天气很好", "action": {"type": "answer", "content": "sunny"}}'
            }]
        },
    ]

    result_mixed = evaluate_reasoning_quality(
        history=mixed_lang_history,
        question="What is the weather like?",
        answer="sunny",
        ground_truth="sunny",
        question_type="FF"
    )
    print(f"Mixed language result: {json.dumps(result_mixed, indent=2, ensure_ascii=False)}\n")


    print("=" * 60)
    print("Test 6: clean_history_media (media cleanup)")
    print("=" * 60)

    original_history = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "http://example.com/img.jpg"},
                {"type": "video", "video": "http://example.com/vid.mp4"},
                {"type": "audio", "audio": "http://example.com/aud.wav"},
                {"type": "text", "text": "This is text content"}
            ]
        }
    ]

    cleaned = clean_history_media(original_history)

    print("Original content types:", [p["type"] for p in original_history[0]["content"]])
    print("Cleaned content types:", [p["type"] for p in cleaned[0]["content"]])
    print("Cleaned text:", cleaned[0]["content"][0]["text"])
