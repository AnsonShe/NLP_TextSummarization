import openai

openai.api_key = "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"

sys_prompt = "Generate a summary of the given article, not exceeding 140 characters."

input_file_path = "test/test.source"
output_file_path = "result/test.out_openai"

# Read articles from the input file
with open(input_file_path, "r", encoding="utf-8") as input_file:
    articles = input_file.readlines()

# Generate summaries and write them to the output file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for article in articles:
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": article}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=140  # Set the max tokens to limit the summary length
        )

        summary = response['choices'][0]['message']['content'].strip()
        output_file.write(summary + "\n")

print("Summaries generated and saved to", output_file_path)
