import argparse
import csv
import os

csv.field_size_limit(10_000_000)

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
LEVELS = ["high", "low"]
TRAIN_N = 128
EVAL_N = 16

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "big5_chat_dataset.csv")


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Derive contrastive steering datasets from big5_chat_dataset.csv."
    )
    parser.add_argument("--out", default=HERE)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    by = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by.setdefault((row["trait"], row["level"]), []).append(row)

    for trait in TRAITS:
        for level in LEVELS:
            sub = by[(trait, level)]
            if len(sub) < TRAIN_N + EVAL_N:
                raise SystemExit(
                    f"{trait}/{level}: only {len(sub)} rows, need {TRAIN_N + EVAL_N}"
                )
            train = sub[:TRAIN_N]
            eval_ = sub[TRAIN_N:TRAIN_N + EVAL_N]
            base = os.path.join(args.out, f"{trait}_{level}")
            write_lines(base + "_train_in.txt", [r["train_input"] for r in train])
            write_lines(base + "_train_out.txt", [r["train_output"] for r in train])
            write_lines(base + "_eval_in.txt", [r["train_input"] for r in eval_])
            write_lines(base + "_eval_out.txt", [r["train_output"] for r in eval_])
            print(f"wrote {trait}/{level}: {TRAIN_N} train + {EVAL_N} eval")


if __name__ == "__main__":
    main()
