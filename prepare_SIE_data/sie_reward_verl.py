# """
# Custom reward function module for verl integration.
# Place this file in verl/verl/reward_model/ or import from your custom path.
# """
# import re
# from typing import Dict, Any, List, Union
# import torch


# class SIERewardFunction:
#     """
#     Reward function for Structured In-Context (SIE) reasoning tasks.
    
#     Implements exact match + format bonus reward as described in the paper.
#     """
    
#     def __init__(
#         self,
#         exact_match_reward: float = 1.0,
#         format_bonus: float = 0.1,
#         no_answer_penalty: float = -0.5,
#         normalize_entities: bool = True
#     ):
#         """
#         Args:
#             exact_match_reward: Reward for correct answer
#             format_bonus: Bonus for proper <think> and <answer> tags
#             no_answer_penalty: Penalty when answer tag is missing
#             normalize_entities: Whether to normalize entities before comparison
#         """
#         self.exact_match_reward = exact_match_reward
#         self.format_bonus = format_bonus
#         self.no_answer_penalty = no_answer_penalty
#         self.normalize_entities = normalize_entities
        
#         # Compile regex patterns
#         self.think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
#         self.answer_re = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    
#     def extract_answer(self, text: str) -> str:
#         """Extract answer from <answer>...</answer> tags."""
#         match = self.answer_re.search(text)
#         if match:
#             answer = match.group(1).strip()
#             if self.normalize_entities:
#                 answer = answer.lower().strip()
#             return answer
#         return ""
    
#     def check_format(self, text: str) -> bool:
#         """Check if text has both <think> and <answer> tags."""
#         text_lower = text.lower()
#         return (
#             "<think>" in text_lower and
#             "</think>" in text_lower and
#             "<answer>" in text_lower and
#             "</answer>" in text_lower
#         )
    
#     def compute_reward(
#         self,
#         generated_text: str,
#         ground_truth: str,
#         all_answers: List[str] = None
#     ) -> float:
#         """
#         Compute scalar reward for generated text.
        
#         Args:
#             generated_text: Model generated output
#             ground_truth: Primary ground truth answer
#             all_answers: List of all valid answers (optional)
        
#         Returns:
#             reward: Float reward value
#         """
#         reward = 0.0
        
#         # Extract predicted answer
#         pred_answer = self.extract_answer(generated_text)
        
#         # Normalize ground truth
#         if self.normalize_entities:
#             gt_norm = ground_truth.lower().strip()
#         else:
#             gt_norm = ground_truth.strip()
        
#         # Check exact match
#         if pred_answer and pred_answer == gt_norm:
#             reward += self.exact_match_reward
#         elif pred_answer and all_answers:
#             # Check alternative answers
#             for alt in all_answers:
#                 if self.normalize_entities:
#                     alt_norm = alt.lower().strip()
#                 else:
#                     alt_norm = alt.strip()
#                 if pred_answer == alt_norm:
#                     reward += self.exact_match_reward
#                     break
        
#         # Format bonus
#         if self.check_format(generated_text):
#             reward += self.format_bonus
        
#         # Penalty for missing answer tag
#         if not self.answer_re.search(generated_text) and len(generated_text.strip()) > 0:
#             reward += self.no_answer_penalty
        
#         return reward
    
#     def __call__(
#         self,
#         batch: Dict[str, Any],
#         generated_texts: Union[List[str], str]
#     ) -> Union[torch.Tensor, List[float]]:
#         """
#         Compute rewards for a batch of samples.
        
#         Args:
#             batch: Batch dict with keys:
#                 - reward_model: dict with ground_truth and all_answers
#             generated_texts: Generated text(s) from model
        
#         Returns:
#             rewards: Tensor of rewards or list of floats
#         """
#         if isinstance(generated_texts, str):
#             generated_texts = [generated_texts]
        
#         rewards = []
        
#         # Get ground truth from batch
#         reward_model_info = batch.get("reward_model", {})
        
#         for text in generated_texts:
#             # Extract ground truth for this sample
#             if isinstance(reward_model_info, dict):
#                 gt = reward_model_info.get("ground_truth", "")
#                 all_ans = reward_model_info.get("all_answers", [])
#             elif isinstance(reward_model_info, list):
#                 # Batch of samples
#                 idx = len(rewards) % len(reward_model_info)
#                 gt = reward_model_info[idx].get("ground_truth", "")
#                 all_ans = reward_model_info[idx].get("all_answers", [])
#             else:
#                 gt = ""
#                 all_ans = []
            
#             reward = self.compute_reward(text, gt, all_ans)
#             rewards.append(reward)
        
#         return torch.tensor(rewards, dtype=torch.float32)


# # Factory function for verl
# def create_sie_reward_fn(**kwargs):
#     """Create and return SIE reward function instance."""
#     return SIERewardFunction(**kwargs)


# # For direct import in verl config
# def sie_reward_function(batch, generated_texts):
#     """
#     Standalone reward function for verl GRPO.
    
#     Args:
#         batch: Batch of data samples
#         generated_texts: List of generated texts
    
#     Returns:
#         rewards: Tensor of reward values
#     """
#     reward_fn = SIERewardFunction()
#     return reward_fn(batch, generated_texts)


# if __name__ == "__main__":
#     # Test the reward function
#     print("Testing SIE Reward Function")
#     print("=" * 50)
    
#     reward_fn = SIERewardFunction()
    
#     test_cases = [
#         {
#             "name": "Correct answer with format",
#             "text": "<think>Analysis here</think>\n<answer>m.person1</answer>",
#             "gt": "m.person1",
#             "expected": 1.1
#         },
#         {
#             "name": "Correct (case insensitive)",
#             "text": "<think>Analysis</think>\n<answer>M.PERSON1</answer>",
#             "gt": "m.person1",
#             "expected": 1.1
#         },
#         {
#             "name": "Wrong answer with format",
#             "text": "<think>Analysis</think>\n<answer>m.wrong</answer>",
#             "gt": "m.person1",
#             "expected": 0.1
#         },
#         {
#             "name": "No format, no answer",
#             "text": "The answer is m.person1",
#             "gt": "m.person1",
#             "expected": -0.5
#         },
#         {
#             "name": "Alternative answer match",
#             "text": "<think>Analysis</think>\n<answer>m.alt</answer>",
#             "gt": "m.person1",
#             "all_ans": ["m.person1", "m.alt"],
#             "expected": 1.1
#         }
#     ]
    
#     for tc in test_cases:
#         reward = reward_fn.compute_reward(
#             tc["text"],
#             tc["gt"],
#             tc.get("all_ans", [])
#         )
#         status = "✓" if abs(reward - tc["expected"]) < 0.01 else "✗"
#         print(f"{status} {tc['name']}: {reward:.2f} (expected {tc['expected']:.2f})")
    
#     print("\n" + "=" * 50)
#     print("Batch test:")
    
#     batch = {
#         "reward_model": {
#             "ground_truth": "m.person1",
#             "all_answers": ["m.person1"]
#         }
#     }
    
#     texts = [
#         "<think>Thinking</think>\n<answer>m.person1</answer>",
#         "<think>Thinking</think>\n<answer>m.wrong</answer>",
#     ]
    
#     rewards = reward_fn(batch, texts)
#     print(f"Rewards: {rewards.tolist()}")
#     print(f"Expected: [1.1, 0.1]")


# """
# Custom reward function module for verl integration.
# Can be used in two ways:
# 1) old/batch style: sie_reward_function(batch, generated_texts)
# 2) new/PR style:   my_sie_reward(data_source, solution_str, ground_truth, extra_info=None)
# """

# import re
# from typing import Dict, Any, List, Union, Optional
# import torch


# class SIERewardFunction:
#     """
#     Reward function for Structured In-Context (SIE) reasoning tasks.

#     - 抽 <answer> ... </answer>
#     - 精确匹配给奖励
#     - 有规范 tag 给额外奖励
#     - 没有 answer tag 给惩罚
#     """

#     def __init__(
#         self,
#         exact_match_reward: float = 1.0,
#         format_bonus: float = 0.1,
#         no_answer_penalty: float = -0.5,
#         normalize_entities: bool = True,
#     ):
#         self.exact_match_reward = exact_match_reward
#         self.format_bonus = format_bonus
#         self.no_answer_penalty = no_answer_penalty
#         self.normalize_entities = normalize_entities

#         self.think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
#         self.answer_re = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

#     # ----------------- small utils ----------------- #
#     def extract_answer(self, text: str) -> str:
#         match = self.answer_re.search(text or "")
#         if match:
#             ans = match.group(1).strip()
#             if self.normalize_entities:
#                 ans = ans.lower().strip()
#             return ans
#         return ""

#     def check_format(self, text: str) -> bool:
#         text_lower = (text or "").lower()
#         return (
#             "<think>" in text_lower
#             and "</think>" in text_lower
#             and "<answer>" in text_lower
#             and "</answer>" in text_lower
#         )

#     # ----------------- core reward ----------------- #
#     def compute_reward(
#         self,
#         generated_text: str,
#         ground_truth: str,
#         all_answers: Optional[List[str]] = None,
#     ) -> float:
#         reward = 0.0

#         pred_answer = self.extract_answer(generated_text)

#         # 归一化 GT
#         if self.normalize_entities:
#             gt_norm = (ground_truth or "").lower().strip()
#         else:
#             gt_norm = (ground_truth or "").strip()

#         # 1) 精确匹配
#         if pred_answer and pred_answer == gt_norm:
#             reward += self.exact_match_reward
#         elif pred_answer and all_answers:
#             for alt in all_answers:
#                 if self.normalize_entities:
#                     alt_norm = str(alt).lower().strip()
#                 else:
#                     alt_norm = str(alt).strip()
#                 if pred_answer == alt_norm:
#                     reward += self.exact_match_reward
#                     break

#         # 2) 格式奖励
#         if self.check_format(generated_text):
#             reward += self.format_bonus

#         # 3) 没有 answer tag 的惩罚
#         if not self.answer_re.search(generated_text or "") and (generated_text or "").strip():
#             reward += self.no_answer_penalty

#         return reward

#     # ----------------- old/batch style ----------------- #
#     def __call__(
#         self,
#         batch: Dict[str, Any],
#         generated_texts: Union[List[str], str],
#     ) -> torch.Tensor:
#         """
#         原来你写的 batch 风格：verl 给你一个 batch dict + N 个生成，你返回 N 个 reward
#         这里保留不动，方便你以前的代码调用
#         """
#         if isinstance(generated_texts, str):
#             generated_texts = [generated_texts]

#         rewards: List[float] = []
#         reward_model_info = batch.get("reward_model", {})

#         for text in generated_texts:
#             if isinstance(reward_model_info, dict):
#                 gt = reward_model_info.get("ground_truth", "")
#                 all_ans = reward_model_info.get("all_answers", [])
#             elif isinstance(reward_model_info, list):
#                 idx = len(rewards) % len(reward_model_info)
#                 gt = reward_model_info[idx].get("ground_truth", "")
#                 all_ans = reward_model_info[idx].get("all_answers", [])
#             else:
#                 gt = ""
#                 all_ans = []

#             r = self.compute_reward(text, gt, all_ans)
#             rewards.append(r)

#         return torch.tensor(rewards, dtype=torch.float32)


# # =============== 旧接口：给你现在的 verl 配置用的 =============== #
# def create_sie_reward_fn(**kwargs):
#     return SIERewardFunction(**kwargs)


# def sie_reward_function(batch, generated_texts):
#     rf = SIERewardFunction()
#     return rf(batch, generated_texts)


# # =============== 新接口：对齐 PR(#452) 的自定义入口 =============== #
# def my_sie_reward(
#     data_source: Dict[str, Any],
#     solution_str: str,
#     ground_truth: str,
#     extra_info: Any = None,
# ) -> float:
#     """
#     这是给 `custom_reward_function.path=...` / `custom_reward_function.name=my_sie_reward`
#     用的单条样本接口。

#     verl 会一条一条调这个函数，所以这里返回 float 就行。
#     """
#     rf = SIERewardFunction()

#     # data_source 里如果已经塞了 all_answers，就拿出来
#     all_answers = None
#     if isinstance(data_source, dict):
#         # 你自己的 SIE step4 里可以把候选答案放到这个 key 下
#         all_answers = data_source.get("all_answers") or data_source.get("answers")

#     return rf.compute_reward(
#         generated_text=solution_str,
#         ground_truth=ground_truth,
#         all_answers=all_answers,
#     )


# if __name__ == "__main__":
#     # 简单自测
#     rf = SIERewardFunction()
#     txt = "<think>ok</think><answer>m.person1</answer>"
#     print(rf.compute_reward(txt, "m.person1"))  # 1.1

#     # 测试 PR 风格
#     ds = {"all_answers": ["m.person1", "m.p1"]}
#     print(my_sie_reward(ds, txt, "m.person1"))  # 1.1








"""
Custom reward function module for verl integration.
Can be used in two ways:
1) old/batch style: sie_reward_function(batch, generated_texts)
2) new/PR style:   my_sie_reward(data_source, solution_str, ground_truth, extra_info=None)
"""

import re
from typing import Dict, Any, List, Union, Optional
import torch


class SIERewardFunction:
    """
    Reward for SIE-style reasoning:

    - <answer>...</answer> 里抽答案
    - 优先按 mid / 别名匹配
    - 答对给 1.0
    - 只有格式给 0.05
    - 没 answer 给 -0.5
    """

    def __init__(
        self,
        exact_match_reward: float = 1.0,
        format_bonus: float = 0.05,     # ① 格式奖励变小
        no_answer_penalty: float = -0.5,
        normalize_entities: bool = True,
    ):
        self.exact_match_reward = exact_match_reward
        self.format_bonus = format_bonus
        self.no_answer_penalty = no_answer_penalty
        self.normalize_entities = normalize_entities

        self.think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
        self.answer_re = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

    # ----------------- utils ----------------- #
    def extract_answer(self, text: str) -> str:
        match = self.answer_re.search(text or "")
        if match:
            ans = match.group(1).strip()
            if self.normalize_entities:
                ans = ans.lower().strip()
            return ans
        return ""

    def check_format(self, text: str) -> bool:
        text_lower = (text or "").lower()
        return (
            "<think>" in text_lower
            and "</think>" in text_lower
            and "<answer>" in text_lower
            and "</answer>" in text_lower
        )

    # 一个小工具：把字符串里的 mid 都抽出来，方便和 GT 里的 mid 比
    # 比如模型输出 <answer>m.04sg82m</answer> 或者 “The answer is m.04sg82m”
    MID_RE = re.compile(r"(m\.[A-Za-z0-9_]+)")

    def extract_mids_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        return self.MID_RE.findall(text)

    # ----------------- core reward ----------------- #
    def compute_reward(
        self,
        generated_text: str,
        ground_truth: str,
        all_answers: Optional[List[str]] = None,
        # 新增：样本里可能给了 mid / 多个 mid
        answer_mids: Optional[List[str]] = None,
    ) -> float:
        """
        answer_mids: 来自数据的标准 mid 列表，比如 ["m.04sg82m"]。
        只要模型输出里出现任意一个，就当答对。
        """
        reward = 0.0

        pred_answer = self.extract_answer(generated_text)

        # 1) 先看有没有 answer_mids，可以直接用 mid 对 mid
        matched_by_mid = False
        if answer_mids:
            # 模型输出里所有 mid
            pred_mids = set(self.extract_mids_from_text(generated_text))
            # 标准答案里的 mid 做小写对齐
            norm_gt_mids = {m.lower().strip() for m in answer_mids}
            # 交集不空就成功
            if any(m.lower().strip() in norm_gt_mids for m in pred_mids):
                matched_by_mid = True
                reward += self.exact_match_reward

        # 2) 没有用 mid 成功的话，再走你原来的“文本精确匹配”
        if not matched_by_mid:
            # 归一化 GT
            if self.normalize_entities:
                gt_norm = (ground_truth or "").lower().strip()
            else:
                gt_norm = (ground_truth or "").strip()

            # 文本精确匹配
            if pred_answer and pred_answer == gt_norm:
                reward += self.exact_match_reward
            elif pred_answer and all_answers:
                for alt in all_answers:
                    if self.normalize_entities:
                        alt_norm = str(alt).lower().strip()
                    else:
                        alt_norm = str(alt).strip()
                    if pred_answer == alt_norm:
                        reward += self.exact_match_reward
                        break
            # 注意：这里不再像你原版那样“答错但格式对也能拿到 0.1”
            # 答错就是 0，下面只给一个很小的格式奖励

        # 3) 格式奖励（缩到 0.05）
        if self.check_format(generated_text):
            reward += self.format_bonus

        # 4) 没有 answer tag 的惩罚
        if not self.answer_re.search(generated_text or "") and (generated_text or "").strip():
            reward += self.no_answer_penalty

        return reward

    # ----------------- old/batch style ----------------- #
    def __call__(
        self,
        batch: Dict[str, Any],
        generated_texts: Union[List[str], str],
    ) -> torch.Tensor:
        if isinstance(generated_texts, str):
            generated_texts = [generated_texts]

        rewards: List[float] = []
        reward_model_info = batch.get("reward_model", {})

        for text in generated_texts:
            if isinstance(reward_model_info, dict):
                gt = reward_model_info.get("ground_truth", "")
                all_ans = reward_model_info.get("all_answers", [])
                # 这里把 mid 也取出来
                answer_mids = (
                    reward_model_info.get("answer_mids")
                    or reward_model_info.get("answer_mid")
                    or reward_model_info.get("mids")
                )
                if isinstance(answer_mids, str):
                    answer_mids = [answer_mids]
            elif isinstance(reward_model_info, list):
                idx = len(rewards) % len(reward_model_info)
                item = reward_model_info[idx]
                gt = item.get("ground_truth", "")
                all_ans = item.get("all_answers", [])
                answer_mids = (
                    item.get("answer_mids")
                    or item.get("answer_mid")
                    or item.get("mids")
                )
                if isinstance(answer_mids, str):
                    answer_mids = [answer_mids]
            else:
                gt = ""
                all_ans = []
                answer_mids = None

            r = self.compute_reward(
                generated_text=text,
                ground_truth=gt,
                all_answers=all_ans,
                answer_mids=answer_mids,
            )
            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32)


# 旧接口
def create_sie_reward_fn(**kwargs):
    return SIERewardFunction(**kwargs)


def sie_reward_function(batch, generated_texts):
    rf = SIERewardFunction()
    return rf(batch, generated_texts)


# PR(#452) 风格单条接口
def my_sie_reward(
    data_source: Any,          # 不强制是 dict 了
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
) -> float:
    """
    给 verl 的单条样本 reward 接口。
    会兼容几种常见输入：
      1) data_source 是 dict，里面直接有 all_answers / answers
      2) data_source 是 dict，里面再包了一层 reward_model
      3) data_source 是 str（比如原始问题），那就当没有候选答案
    """
    rf = SIERewardFunction()

    all_answers = None

    # 情况 1：真正的字典
    if isinstance(data_source, dict):
        # 直接放在顶层的
        all_answers = (
            data_source.get("all_answers")
            or data_source.get("answers")
        )

        # 有些数据会这样放：{"reward_model": {"ground_truth": ..., "all_answers": [...]}}
        if all_answers is None and "reward_model" in data_source:
            rm = data_source["reward_model"]
            if isinstance(rm, dict):
                all_answers = rm.get("all_answers") or rm.get("answers")

    # 情况 2：如果是 list，你也可以取第一个作为兜底
    elif isinstance(data_source, list) and data_source:
        first = data_source[0]
        if isinstance(first, dict):
            all_answers = (
                first.get("all_answers")
                or first.get("answers")
            )

    # 其他情况（str / None）就当没有候选答案
    return rf.compute_reward(
        generated_text=solution_str,
        ground_truth=ground_truth,
        all_answers=all_answers,
    )