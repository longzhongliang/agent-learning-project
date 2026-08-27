"""RAGAS 评估脚本：量化 RAG 检索质量。

指标含义：
- faithfulness（忠实度）：回答有没有编造知识库没有的内容（越高越好）
- answer_relevancy（答案相关性）：回答有没有答非所问（越高越好）
- context_precision（上下文精度）：检索到的片段是不是真有用（越高越好）

前置条件：
1. 安装依赖：pip install ragas datasets
2. 配置 DEEPSEEK_API_KEY（评估需要一个 LLM 当"裁判"）
3. 知识库目录已有文档（workspace/knowledge/）

运行：python scripts/evaluate_rag.py
"""

from __future__ import annotations

import os
import re

# 1. 依赖检查（ragas 未安装时给出安装提示）
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
except ImportError:
    print("缺少 ragas 或 datasets，请先安装：")
    print("  pip install ragas datasets")
    raise SystemExit(1)

from langchain_deepseek import ChatDeepSeek

from app.config import load_settings
from app.rag import KnowledgeBase, build_rag_paths

# ========== 测试问题集（改成你自己的） ==========
# 每个问题配一个标准答案（ground_truth），用于评估"答得对不对"
TEST_CASES = [
    {
        "question": "保修期是多久？",
        "ground_truth": "整机保修一年，电池保修六个月，人为损坏不在保修范围内。",
    },
    {
        "question": "智能音箱Pro多少钱？",
        "ground_truth": "智能音箱Pro 价格 399 元。",
    },
    {
        "question": "蓝牙连不上怎么办？",
        "ground_truth": "先关闭其他设备的蓝牙占用，然后重置音箱。",
    },
]


def extract_chunk_texts(search_output: str) -> list[str]:
    """从 search() 的输出里抽出每个片段的正文（去掉【片段头】）。"""
    blocks = re.split(r"【片段 \d+｜.*?】\n", search_output)
    return [block.strip() for block in blocks if block.strip()]


def main() -> None:
    settings = load_settings()
    kb = KnowledgeBase(build_rag_paths(settings))

    # 2. 用 DeepSeek 当"回答者"和"裁判"
    model = ChatDeepSeek(model=settings.model_name, temperature=0)

    questions, answers, contexts_list, ground_truths = [], [], [], []
    for case in TEST_CASES:
        question = case["question"]
        search_output = kb.search(question, k=3)
        contexts = extract_chunk_texts(search_output)

        if not contexts:
            print(f"[跳过] 问题「{question}」未检索到内容，无法评估")
            continue

        # 回答者：把检索结果喂给 LLM，让它基于片段作答（模拟生产流程）
        prompt = (
            f"根据以下知识库片段回答问题。只依据片段内容，不要编造。\n\n"
            f"片段：\n{search_output}\n\n问题：{question}"
        )
        answer = model.invoke(prompt).content

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(case["ground_truth"])
        print(f"[已评估] {question}")

    if not questions:
        print("没有可评估的问题。")
        return

    # 3. 组装数据集并评估
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\n========== RAGAS 评估结果 ==========")
    for metric, score in result.items():
        print(f"{metric}: {score:.4f}")
    print("===================================")
    print("解读：faithfulness/answer_relevancy/context_precision 都越接近 1 越好。")


if __name__ == "__main__":
    main()
