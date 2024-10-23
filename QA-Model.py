import argparse
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def print_sample_processed_squad_entry(processed_squad_data, title_to_id_mapping, context_to_id_mapping, index):

    entry = processed_squad_data[index]
    title = title_to_id_mapping.get(entry["title_id"], "Unknown Title")
    context = context_to_id_mapping.get(entry["context_id"], "Unknown Context")

    print(f"Sample Entry at Index {index}:")
    print(f"Title: {title}")
    print(f"Context: {context}")
    print(f"Question: {entry['question']}")
    print(f"Answers: {entry['answers']['text']}")
    print(f"Answer Start Positions: {entry['answers']['answer_start']}")
    print(f"Answer End Positions: {entry['answers']['answer_end']}")

def process_raw_squad_data(raw_squad_data):
    processed_squad_data = []
    title_to_ids_mapping = {}
    context_to_ids_mapping = {}

    for article in raw_squad_data["data"]:
        title = article.get("title", "").strip()
        title_id = len(title_to_ids_mapping)
        title_to_ids_mapping[title_id] = title

        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            context_id = len(context_to_ids_mapping)
            context_to_ids_mapping[context_id] = context

            for qa_pair in paragraph["qas"]:
                question = qa_pair["question"].strip()

                # Calculate answer starts and ends
                answer_starts = [answer["answer_start"] for answer in qa_pair["answers"]]
                answers = [answer["text"].strip() for answer in qa_pair["answers"]]
                answer_ends = [start + len(text) for start, text in zip(answer_starts, answers)]

                processed_squad_data.append({
                    "title_id": title_id,
                    "context_id": context_id,
                    "question": question,
                    "answers": {
                        "answer_start": answer_starts,
                        "answer_end": answer_ends,
                        "text": answers
                    }
                })

    return processed_squad_data, title_to_ids_mapping, context_to_ids_mapping


def main(args):
    raw_squad_data = read_json_file(args.train_json_path)

    proccessed_squad_data, title_to_ids_mapping, context_to_ids_mapping = process_raw_squad_data(raw_squad_data)

    print_sample_processed_squad_entry(proccessed_squad_data, title_to_ids_mapping, context_to_ids_mapping, 0)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json_path', help='path to the json training file')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)

# python3 QA-Model.py --train_json_path train-v1.1.json
