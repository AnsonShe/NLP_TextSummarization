import argparse
from rouge import Rouge

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def calculate_rouge(hypothesis_path, reference_path):
    hypothesis_list = read_file(hypothesis_path)
    reference_list = read_file(reference_path)

    rouge = Rouge()

    total_scores = {"rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
                    "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}}

    num_samples = len(hypothesis_list)

    for hyp, ref in zip(hypothesis_list, reference_list):
        scores = rouge.get_scores(hyp, ref)[0]
        for rouge_type in total_scores.keys():
            for metric in ["f", "p", "r"]:
                total_scores[rouge_type][metric] += scores[rouge_type][metric]

    # Calculate averages
    for rouge_type in total_scores.keys():
        for metric in ["f", "p", "r"]:
            total_scores[rouge_type][metric] /= num_samples

    return total_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average ROUGE scores.")
    parser.add_argument("--hyp", required=True, help="Path to the hypothesis file.")
    parser.add_argument("--ref", required=True, help="Path to the reference file.")
    args = parser.parse_args()

    scores = calculate_rouge(args.hyp, args.ref)
    print("Average ROUGE Scores:")
    print(scores)
