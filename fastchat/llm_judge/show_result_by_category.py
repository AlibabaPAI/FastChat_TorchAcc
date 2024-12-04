"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
import json

def load_vocabory():
    """Load questions from a file."""
    question_file = f"data/mt_bench/question.jsonl"
    vocabory = {}

    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                question = json.loads(line)
                vocabory[question['question_id']] = question['category']
    return vocabory

def display_result_single(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    vocabory = load_vocabory()

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["question_id", "model", "score", "turn"]]
    df = df[df["score"] != -1]
    df['vocabory'] = [vocabory[qid] for qid in df['question_id']]
    df = df[["vocabory", "model", "score", "turn"]]

    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "vocabory", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "vocabory", "turn"]).mean()
        print(df_2.sort_values(by="vocabory", ascending=False))

        print("\n########## Average by vocabory##########")
        df_3 = df[["model", "vocabory", "score"]].groupby(["model", "vocabory"]).mean()
        print(df_3.sort_values(by="vocabory", ascending=False))

        print("\n########## Average ##########")
        df_4 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_4.sort_values(by="score", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    args = parser.parse_args()

    display_result_single(args)
