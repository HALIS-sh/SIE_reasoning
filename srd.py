# srd_distill_and_filter_transformers.py
# 将 SIE 样本 (Q, SI, target) 转为 SRD：(<think>…</think>, <answer>…</answer>)
# 纯本地版（Transformers 直连）。不需要也不会调用任何 HTTP API。

import json, re, random, os, time, sys, unicodedata
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

THINK_S, THINK_E = "<think>", "</think>"
ANS_S, ANS_E = "<answer>", "</answer>"

# =========================
# 配置
# =========================
@dataclass
class DistillCfg:
    # 数据与输出路径
    sie_jsonl: str = "/home/jyzhou/toy_project/SIE_reasoning/prepare_SIE_data/data/RoG-webqsp-step3-step4-sie100.jsonl"
    out_srd_jsonl: str = "/home/jyzhou/toy_project/SIE_reasoning/baseline_SRD/srd_data/RoG-webqsp-SRD-LOCAL-TF.jsonl"

    # 采样与随机性
    k_samples: int = 3
    max_new_tokens: int = 1024
    seed: int = 42

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.05

    # 本地 Transformers 模型
    hf_model_name_or_path: str = "Qwen/QwQ-32B"  # 使用你示例中的模型名
    hf_device: str = "auto"      
    hf_dtype: str = "auto"      
    hf_use_chat_template: bool = True   # 有模板就用
    hf_local_files_only: bool = False   # 需从 Hub 拉取则设为 False；离线改 True 并放到本地目录
    hf_low_cpu_mem_usage: bool = True

    # （可选）量化：需要 bitsandbytes，对纯 FP 运行可忽略（32B 建议 4bit）
    load_in_8bit: bool = False
    load_in_4bit: bool = False  # 关闭 4bit，避免 bitsandbytes 缺失错误；需要可再开启

    # 新增：是否使用 pipeline 统一接口
    use_pipeline: bool = True

    # 评估/日志
    print_every: int = 5
    keep_reject_log: bool = True
    reject_log_path: Optional[str] = None

    # 匹配策略
    use_loose_fallback: bool = True
    accept_substring: bool = True

    # 提示词控制
    use_existing_prompt: bool = False

    # 调试
    debug_dump_first: int = 3
    process_limit: Optional[int] = 20  # 跑全量设为 None

# =========================
# 文本规范化与匹配
# =========================
_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_TAGS_RE = re.compile(r"</?\s*(?:think|answer)\s*>", re.I)
_FINAL_ANSWER_PAT = re.compile(
    r"(?:^|\n|\r)\s*(?:final\s*answer|answer|最终答案|最后答案|答案|结论)\s*[:：]\s*([^\n\r]+)",
    re.IGNORECASE
)

def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def norm_basic(s: str) -> str:
    s = s.strip().lower()
    s = _WHITESPACE.sub(" ", s)
    return s

def norm_qa(s: str) -> str:
    s = norm_basic(s)
    s = strip_accents(s)
    s = _PUNCT.sub("", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s).strip()
    s = _WHITESPACE.sub(" ", s)
    return s

def em_strict(pred: str, target: str) -> bool:
    return norm_basic(pred) == norm_basic(target)

def em_loose(pred: str, target: str, accept_substring: bool = True) -> bool:
    p = norm_qa(pred); t = norm_qa(target)
    if p == t: return True
    if accept_substring: return (p in t) or (t in p)
    return False

def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    s = text.find(start_tag)
    if s < 0: return None
    e = text.find(end_tag, s + len(start_tag))
    if e < 0: return None
    return text[s + len(start_tag):e].strip()

def validate_response_structure(txt: str) -> Tuple[bool, bool]:
    begin_valid = txt.lstrip().startswith(THINK_S)
    tags = {
        "think_start": (THINK_S, 1),
        "think_end": (THINK_E, 1),
        "answer_start": (ANS_S, 1),
        "answer_end": (ANS_E, 1),
    }
    positions, ok = {}, True
    for name, (tag, cnt) in tags.items():
        c = txt.count(tag)
        positions[name] = txt.find(tag)
        if c != cnt:
            ok = False
    if (positions["think_start"] > positions["think_end"] or
        positions["think_end"] > positions["answer_start"] or
        positions["answer_start"] > positions["answer_end"]):
        ok = False
    return ok, begin_valid

def sanitize_tags(s: Optional[str]) -> str:
    if not s: return ""
    return _TAGS_RE.sub("", s).strip()

def split_cot_and_ans(generated: str) -> Tuple[str, str]:
    """
    从模型单段生成中分离 COT 与答案：
    1) 优先找 'Final answer: xxx'（取最后一次匹配）；
    2) 否则用最后一行非空作答案，其前面为 COT；
    3) 清理潜在的 <think>/<answer> 标签。
    """
    if not generated:
        return "", ""
    text = sanitize_tags(generated)
    matches = list(_FINAL_ANSWER_PAT.finditer(text))
    if matches:
        m = matches[-1]
        ans = m.group(1).strip()
        cot = text[:m.start()].strip()
        return cot, ans
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines: return "", ""
    ans = lines[-1]
    cot = "\n".join(lines[:-1])
    return cot, ans

# =========================
# Gold 抽取
# =========================
def _first_non_empty_str(x: Union[str, List, Dict]) -> Optional[str]:
    if x is None: return None
    if isinstance(x, str): return x.strip() or None
    if isinstance(x, list):
        for v in x:
            if isinstance(v, str) and v.strip():
                return v.strip()
    if isinstance(x, dict):
        for k in ("text", "name", "string", "value"):
            if k in x and isinstance(x[k], str) and x[k].strip():
                return x[k].strip()
    return None

def extract_ground_truth(obj: Dict[str, Any]) -> Optional[str]:
    rm = obj.get("reward_model")
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if isinstance(gt, dict):
            val = _first_non_empty_str(gt.get("target"))
            if val: return val
    gt = obj.get("ground_truth")
    if isinstance(gt, dict):
        val = _first_non_empty_str(gt.get("target"))
        if val: return val
    for k in ("target", "answer", "answers", "gold", "label", "labels"):
        if k in obj:
            val = _first_non_empty_str(obj[k])
            if val: return val
    extra = obj.get("extra_info")
    if isinstance(extra, dict):
        for k in ("target", "answer", "answers", "gold", "label", "labels"):
            if k in extra:
                val = _first_non_empty_str(extra[k])
                if val: return val
    return None

# =========================
# Structured context 格式化
# =========================
def format_structured_context(si_obj: Any) -> str:
    """
    将 structured_context（或 triples）格式化成更容易读的文本。
    """
    if isinstance(si_obj, str):
        return si_obj
    if isinstance(si_obj, list):
        lines = []
        for it in si_obj:
            if isinstance(it, (list, tuple)) and len(it) == 3:
                h, r, t = it
                lines.append(f"- ({h}) -[{r}]-> ({t})")
            else:
                lines.append(f"- {it}")
        return "\n".join(lines)
    try:
        return json.dumps(si_obj, ensure_ascii=False)
    except Exception:
        return str(si_obj)

# =========================
# Prompt（不诱导标签，要求以 Final answer 收尾）
# =========================
def build_user_prompt(question: str, structured_context: str) -> str:
    return (
        "You are a careful reasoner. Use ONLY the provided structured context (SIE) as soft evidence.\n"
        "Think step by step, but DO NOT print any <think> or <answer> tags.\n"
        "At the end, output a single short line starting with 'Final answer: ' followed by ONLY the answer text.\n\n"
        "[Structured context | SIE]\n"
        f"{structured_context}\n\n"
        "[Question]\n"
        f"{question}\n"
    )

# =========================
# 本地 Transformers 模型
# =========================
class HFLocalModel:
    def __init__(self, cfg: DistillCfg):
        self.cfg = cfg
        self.tok = None
        self.mdl = None
        self._load()

    def _select_dtype(self, torch):
        if self.cfg.hf_dtype != "auto":
            return getattr(torch, self.cfg.hf_dtype)
        # auto：优先 bfloat16（A100/H100/近年显卡），否则 float16；CPU 则 float32
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.float16
        return torch.float32

    def _load(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            dtype = self._select_dtype(torch)

            quantization_config = None
            if (self.cfg.load_in_4bit or self.cfg.load_in_8bit):
                try:
                    from transformers import BitsAndBytesConfig
                except Exception:
                    # bitsandbytes 不可用则关闭量化
                    self.cfg.load_in_4bit = False
                    self.cfg.load_in_8bit = False
                if self.cfg.load_in_4bit:
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
                elif self.cfg.load_in_8bit:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            self.tok = AutoTokenizer.from_pretrained(
                self.cfg.hf_model_name_or_path,
                trust_remote_code=True,
                local_files_only=self.cfg.hf_local_files_only
            )
            self.mdl = AutoModelForCausalLM.from_pretrained(
                self.cfg.hf_model_name_or_path,
                device_map=self.cfg.hf_device if self.cfg.hf_device != "auto" else "auto",
                torch_dtype=None if quantization_config else dtype,
                quantization_config=quantization_config,
                trust_remote_code=True,
                local_files_only=self.cfg.hf_local_files_only,
                low_cpu_mem_usage=self.cfg.hf_low_cpu_mem_usage
            )
            if self.cfg.use_pipeline:
                self.pipe = pipeline(
                    "text-generation",
                    model=self.mdl,
                    tokenizer=self.tok,
                )
            else:
                self.pipe = None

            # 种子（影响采样）
            try:
                torch.manual_seed(self.cfg.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.cfg.seed)
            except Exception:
                pass

        except Exception as e:
            raise RuntimeError(f"加载本地模型失败：{e}")

    def generate_once(self, prompt: str) -> str:
        # 使用 pipeline 路径
        if getattr(self, "pipe", None) is not None:
            messages = [{"role": "user", "content": prompt}]
            # Qwen 系列 chat 模型支持直接传 messages 给 pipeline
            out = self.pipe(messages, max_new_tokens=self.cfg.max_new_tokens, temperature=self.cfg.temperature, top_p=self.cfg.top_p)
            # pipeline 返回列表
            text = out[0]["generated_text"] if isinstance(out, list) else str(out)
            return text.strip()

        import torch
        tok, mdl = self.tok, self.mdl

        messages = [
            {"role": "system", "content": "You are a helpful, careful reasoner."},
            {"role": "user", "content": prompt}
        ]
        if self.cfg.hf_use_chat_template and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        else:
            stitched = "System: You are a helpful, careful reasoner.\nUser:\n" + prompt + "\nAssistant:"
            inputs = tok(stitched, return_tensors="pt").input_ids
        if hasattr(mdl, "device") and str(mdl.device) != "cpu":
            inputs = inputs.to(mdl.device)
        with torch.no_grad():
            gen_out = mdl.generate(
                inputs,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=self.cfg.repetition_penalty,
                use_cache=True,
            )
        try:
            gen_only = gen_out[0][inputs.shape[-1]:]
            text = tok.decode(gen_only, skip_special_tokens=True)
        except Exception:
            text = tok.decode(gen_out[0], skip_special_tokens=True)
            if self.cfg.hf_use_chat_template and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                cut = text.rfind("Assistant:")
                if cut != -1:
                    text = text[cut + len("Assistant:"):].strip()
        return text.strip()

# =========================
# 主流程
# =========================
def build_prompt_from_obj(obj: Dict[str, Any]) -> str:
    question = obj.get("question", "")
    si = obj.get("structured_context") or obj.get("si", "")
    if not si and obj.get("triples"):
        si = [(h, r, t) for h, r, t in obj["triples"]]
    si_txt = format_structured_context(si)
    return build_user_prompt(question, si_txt)

def main():
    cfg = DistillCfg()
    random.seed(cfg.seed)

    os.makedirs(os.path.dirname(cfg.out_srd_jsonl), exist_ok=True)
    rej_path = cfg.reject_log_path or (cfg.out_srd_jsonl + ".reject.log")
    rej_f = open(rej_path, "w", encoding="utf-8") if cfg.keep_reject_log else None

    kept = total = 0
    miss_gt = em_fail = fmt_fail = api_fail = 0

    model = HFLocalModel(cfg)

    def _trunc(s: Optional[str], n: int = 400) -> str:
        if not s: return ""
        return s[:n] + (" …" if len(s) > n else "")

    with open(cfg.sie_jsonl, "r", encoding="utf-8") as fin, open(cfg.out_srd_jsonl, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, 1):
            if cfg.process_limit is not None and idx > cfg.process_limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)

            # 1) Prompt
            if cfg.use_existing_prompt and isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
                q = obj["prompt"]
            else:
                q = build_prompt_from_obj(obj)

            # 2) Gold
            gt = extract_ground_truth(obj)
            if not gt:
                miss_gt += 1; total += 1
                if rej_f:
                    rej_f.write(json.dumps({"idx": idx, "reason": "no_gold", "obj_keys": list(obj.keys())}, ensure_ascii=False) + "\n")
                continue

            # 3) 采样（本地）
            candidates: List[str] = []
            for _ in range(cfg.k_samples):
                text = model.generate_once(q)
                cot, ans = split_cot_and_ans(text)
                full = f"{THINK_S}{cot}{THINK_E}\n{ANS_S}{ans}{ANS_E}"
                candidates.append(full)
                time.sleep(0.02)

            # 4) 调试
            if idx <= cfg.debug_dump_first:
                sys.stdout.write(f"\n=== DEBUG SAMPLE {idx} ===\n")
                sys.stdout.write(f"GT: {gt}\n")
                sys.stdout.write(f"Prompt(head): { _trunc(q, 300) }\n")
                for j, c in enumerate(candidates, 1):
                    a = extract_between(c, ANS_S, ANS_E) if c else None
                    sys.stdout.write(f"cand[{j}] fmt_ok={validate_response_structure(c)[0] if c else False} ans={_trunc(a, 120)}\n")
                sys.stdout.flush()

            # 5) 拒采筛选
            best = None
            for c in candidates:
                if not c: continue
                fmt_ok, _ = validate_response_structure(c)
                ans = extract_between(c, ANS_S, ANS_E)
                if not ans: continue
                if fmt_ok and em_strict(ans, gt):
                    best = c; break

            if best is None and cfg.use_loose_fallback:
                for c in candidates:
                    if not c: continue
                    fmt_ok, _ = validate_response_structure(c)
                    ans = extract_between(c, ANS_S, ANS_E)
                    if not ans: continue
                    if fmt_ok and em_loose(ans, gt, accept_substring=cfg.accept_substring):
                        best = c; break

            total += 1
            if best is None:
                local_fmt_fail = 0
                local_em_fail = 0
                for c in candidates:
                    if not c: continue
                    fmt_ok, _ = validate_response_structure(c)
                    ans = extract_between(c, ANS_S, ANS_E)
                    if not fmt_ok:
                        local_fmt_fail += 1
                    elif ans is None or (not em_strict(ans, gt) and not (cfg.use_loose_fallback and em_loose(ans, gt, cfg.accept_substring))):
                        local_em_fail += 1
                fmt_fail += (local_fmt_fail > 0)
                em_fail += (local_em_fail > 0)
                if rej_f:
                    rej_f.write(json.dumps({"idx": idx, "reason": "reject", "gt": gt, "candidates_preview": candidates[:2]}, ensure_ascii=False) + "\n")
                continue

            think = extract_between(best, THINK_S, THINK_E) or ""
            ans  = extract_between(best, ANS_S, ANS_E) or ""
            rec = {
                "question": obj.get("question", ""),
                "structured_context": obj.get("structured_context", obj.get("si", "")),
                "target": gt,
                "cot": think,
                "answer": ans
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            kept += 1
            if idx % cfg.print_every == 0:
                sys.stdout.write(
                    f"[processed {idx}] kept={kept} total={total} "
                    f"(miss_gt={miss_gt}, api_fail={api_fail}, fmt_fail≈{fmt_fail}, em_fail≈{em_fail})\n"
                )
                sys.stdout.flush()

    if rej_f: rej_f.close()
    print(f"FINISHED: kept {kept} / {total} = {kept / max(1,total):.2%}, "
          f"miss_gt={miss_gt}, api_fail={api_fail}, fmt_fail≈{fmt_fail}, em_fail≈{em_fail}")

if __name__ == "__main__":
    main()
