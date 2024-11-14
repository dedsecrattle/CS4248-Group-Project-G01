import argparse
import json
from pathlib import Path
from datasets import load_dataset

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def print_sample_processed_squad_entry(processed_squad_data_path, index):
    current_directory = Path(__file__).parent

    processed_squad_data = load_json_file(current_directory / processed_squad_data_path)['data']

    entry = processed_squad_data[index]

    print(f"Sample Entry at Index {index}:")
    print(f"Title: {entry['title']}")
    print(f"Context: {entry['context']}")
    print(f"Question: {entry['question']}")
    print(f"Answers: {entry['answers']['text']}")
    print(f"Answer Start Positions: {entry['answers']['answer_start']}")
    print(f"Answer End Positions: {entry['answers']['answer_end']}")

def process_raw_squad_data(raw_squad_data, processed_squad_data_path):
    processed_squad_data = []

    for article in raw_squad_data["data"]:
        title = article.get("title", "").strip()

        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()

            for qa_pair in paragraph["qas"]:
                question = qa_pair["question"].strip()

                # Calculate answer starts and ends
                answer_starts = [answer["answer_start"] for answer in qa_pair["answers"]]
                answers = [answer["text"].strip() for answer in qa_pair["answers"]]
                answer_ends = [start + len(text) for start, text in zip(answer_starts, answers)]

                processed_squad_data.append({
                    "title": title,
                    "context": context,
                    "question": question,
                    "answers": {
                        "answer_start": answer_starts,
                        "answer_end": answer_ends,
                        "text": answers
                    }
                })
    
    current_directory = Path(__file__).parent

    with open(current_directory / processed_squad_data_path, 'w') as file:
      data = {'data': processed_squad_data}
      file.write(json.dumps(data))

def main(args):
    raw_squad_data = load_json_file(args.train_json_path)

    process_raw_squad_data(
        raw_squad_data,
        'processed_squad_data.json',
    )
    
    print_sample_processed_squad_entry(
        'processed_squad_data.json',
        0
    )

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_path', help='path to the json training file')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)

# python3 squad_preprocessing.py --train_json_path train-v1.1.json
