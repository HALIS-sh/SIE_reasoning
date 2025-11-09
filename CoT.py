from transformers import AutoTokenizer
from vllm import LLM  # 如果想用高效推理
import json

model_name = "Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = LLM(model=model_name, device="cuda")  # vllm推理
jsonl_path = "/mnt/sdb/zhangshuang//SII/data/mydata.jsonl"
sie_data = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        question = item["question"]
        answer = item["answer"]
        structured_context = item["structured_context"]  # 直接用这个字段

        sie_data.append((question, structured_context, answer))

def construct_cot_prompt(question, SI):
    return f"<think>\nStep-by-step reasoning based on SI:\n{SI}\n</think>\nQuestion: {question}\n<answer>"
def cot_inference(prompt):
    outputs = llm.generate([prompt], max_tokens=100)
    return outputs[0].outputs[0].text
def compute_accuracy(preds, trues):
    correct = sum([p == t for p, t in zip(preds, trues)])
    return correct / len(trues)
preds, trues = [], []
output_path = "/mnt/sdb/zhangshuang/SII/SII/results/cot_results.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for Q, SI, A_true in sie_data:
        prompt = construct_cot_prompt(Q, SI)
        output_text = cot_inference(prompt)
        answer = output_text.split("<answer>")[1].split("</answer>")[0].strip()
        preds.append(answer)
        trues.append(A_true)
        f.write(f"Q: {Q}\n")
        f.write(f"Predicted Answer: {answer}\n")
        f.write(f"Ground Truth: {A_true}\n")
        f.write("-" * 30 + "\n")

accuracy = compute_accuracy(preds, trues)
print(f"Accuracy: {accuracy:.4f}")