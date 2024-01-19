import argparse
from nltk.tokenize import word_tokenize
import nltk.translate.meteor_score as me

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def preprocess(sentence):
    # Perform any necessary preprocessing on the sentence
    # You can customize this function based on your requirements
    return ' '.join(word_tokenize(sentence.lower()))

def calculate_meteor(hypothesis_path, reference_path):
    hypothesis_list = read_file(hypothesis_path)
    reference_list = read_file(reference_path)

    # Tokenize and preprocess sentences
    hypothesis_tokens = [word_tokenize(preprocess(sentence)) for sentence in hypothesis_list]
    reference_tokens = [word_tokenize(preprocess(sentence)) for sentence in reference_list]

    # Calculate METEOR score for each sample and average
    total_meteor_score = 0
    num_samples = len(hypothesis_tokens)

    for i in range(num_samples):
        meteor_score_value = me.single_meteor_score(set(reference_tokens[i]), set(hypothesis_tokens[i]))
        total_meteor_score += meteor_score_value

    average_meteor_score = total_meteor_score / num_samples

    return average_meteor_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate METEOR scores.")
    parser.add_argument("--hyp", required=True, help="Path to the hypothesis file.")
    parser.add_argument("--ref", required=True, help="Path to the reference file.")
    args = parser.parse_args()

    meteor_score_value = calculate_meteor(args.hyp, args.ref)

    print("METEOR Score:")
    print(f"{meteor_score_value:.6f}")
