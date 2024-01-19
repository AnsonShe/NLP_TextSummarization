import argparse
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def calculate_bleu(hypothesis_path, reference_path):
    hypothesis_list = read_file(hypothesis_path)
    reference_list = read_file(reference_path)

    # Convert each hypothesis and reference into a list of tokens
    hypothesis_tokens = [h.split() for h in hypothesis_list]
    reference_tokens = [[r.split()] for r in reference_list]

    # Calculate corpus-level BLEU
    corpus_bleu_score = corpus_bleu(reference_tokens, hypothesis_tokens, smoothing_function=SmoothingFunction().method3)

    return corpus_bleu_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEU scores.")
    parser.add_argument("--hyp", required=True, help="Path to the hypothesis file.")
    parser.add_argument("--ref", required=True, help="Path to the reference file.")
    args = parser.parse_args()

    corpus_bleu_score = calculate_bleu(args.hyp, args.ref)

    print("Corpus BLEU Score:")
    print(f"{corpus_bleu_score:.6f}")
