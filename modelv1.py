import argparse
import collections
import numpy as np
import torch
from torch import nn
from transformers import (
    BertTokenizerFast,
    BertModel,
    BertPreTrainedModel,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset
import evaluate
from pathlib import Path

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Retro-Reader Training and Inference Script")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        required=True,
        help="Mode to run the script: 'train' or 'inference'",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results",
        help="Path to save or load the model",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question for inference mode",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Context for inference mode",
    )
    args = parser.parse_args()

    # Initialize the tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if args.mode == "train":
        # Training mode
        # Step 1: Load the SQuAD v1.1 dataset

        # Modified to load provided dataset:

        # dataset = load_dataset('squad')

        current_directory = Path(__file__).parent
        processed_squad_data_path = current_directory / "processed_squad_data.json"
        dataset = load_dataset('json', data_files=str(processed_squad_data_path), field='data')

        # Step 2: Preprocess the data
        def preprocess_function(examples):
            questions = [q.lstrip() for q in examples['question']]
            contexts = examples['context']
            answers = examples['answers']

            # Tokenize inputs
            inputs = tokenizer(
                questions,
                contexts,
                max_length=384,
                truncation="only_second",
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # Since one example might give multiple features if it is long,
            # we need to keep track of the example index and offset mappings
            sample_mapping = inputs.pop("overflow_to_sample_mapping")
            offset_mapping = inputs["offset_mapping"]

            # Labels
            inputs["start_positions"] = []
            inputs["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = inputs["input_ids"][i]
                sample_idx = sample_mapping[i]
                answer = answers[sample_idx]
                if len(answer["answer_start"]) == 0:
                    # If there is no answer, set the start and end positions to 0
                    inputs["start_positions"].append(0)
                    inputs["end_positions"].append(0)
                else:
                    # Start/end character index of the answer in the text
                    start_char = answer["answer_start"][0]
                    end_char = start_char + len(answer["text"][0])

                    # Start token index of the current span in the text
                    sequence_ids = inputs.sequence_ids(i)

                    # Find the start and end of the context
                    idx = 0
                    while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                        idx += 1
                    context_start = idx
                    while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                        idx += 1
                    context_end = idx - 1

                    # If the answer is not fully inside the context, label it (0, 0)
                    if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
                        inputs["start_positions"].append(0)
                        inputs["end_positions"].append(0)
                    else:
                        # Otherwise, find the start and end token indices
                        idx = context_start
                        while idx <= context_end and offsets[idx][0] <= start_char:
                            idx += 1
                        inputs["start_positions"].append(idx - 1)

                        idx = context_end
                        while idx >= context_start and offsets[idx][1] >= end_char:
                            idx -= 1
                        inputs["end_positions"].append(idx + 1)

            inputs["example_id"] = [examples["id"][sample_mapping[i]] for i in range(len(sample_mapping))]
            return inputs
        
        # Modified to work with provided dataset: 

        # tokenized_datasets = dataset.map(
        #     preprocess_function, batched=True, remove_columns=dataset["train"].column_names
        # )

        tokenized_datasets = dataset.map(
            preprocess_function, batched=True
        )

        # Step 3: Define the Retro-Reader model with Sketchy and Intensive Readers
        class RetroReader(BertPreTrainedModel):
            def __init__(self, config):
                super(RetroReader, self).__init__(config)
                self.config = config

                # Sketchy Reader Encoder (shared with Intensive Reader for efficiency)
                self.bert = BertModel(config)
                
                # Sketchy Reader Output Layer
                self.sketchy_qas_output = nn.Linear(config.hidden_size, 2)

                # Intensive Reader Encoder (additional layer)
                self.intensive_layer = nn.Linear(config.hidden_size, config.hidden_size)

                # Intensive Reader Output Layer
                self.intensive_qas_output = nn.Linear(config.hidden_size, 2)

                # Answer Verification Module
                self.verifier = nn.Sequential(
                    nn.Linear(2 * config.hidden_size, config.hidden_size),
                    nn.Tanh(),
                    nn.Linear(config.hidden_size, 1),
                    nn.Sigmoid()
                )

                self.init_weights()

            def forward(
                self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                start_positions=None,
                end_positions=None,
            ):
                # Sketchy Reading Stage
                outputs = self.bert(
                    input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
                )  # (batch_size, seq_len, hidden_size)
                sequence_output = outputs[0]

                # Sketchy Reader Start and End logits
                sketchy_logits = self.sketchy_qas_output(sequence_output)
                sketchy_start_logits, sketchy_end_logits = sketchy_logits.split(1, dim=-1)
                sketchy_start_logits = sketchy_start_logits.squeeze(-1)
                sketchy_end_logits = sketchy_end_logits.squeeze(-1)

                # Get top k start and end positions from Sketchy Reader
                k = 5  # Number of top positions to consider
                sketchy_start_prob = torch.softmax(sketchy_start_logits, dim=-1)
                sketchy_end_prob = torch.softmax(sketchy_end_logits, dim=-1)

                # Get the indices of the top k probable positions
                # For simplicity, we process the entire sequence in the Intensive Reader
                # In practice, you could focus on the top-k positions

                # Intensive Reading Stage
                intensive_sequence_output = self.intensive_layer(sequence_output)  # (batch_size, seq_len, hidden_size)

                # Intensive Reader Start and End logits
                intensive_logits = self.intensive_qas_output(intensive_sequence_output)
                intensive_start_logits, intensive_end_logits = intensive_logits.split(1, dim=-1)
                intensive_start_logits = intensive_start_logits.squeeze(-1)
                intensive_end_logits = intensive_end_logits.squeeze(-1)

                # Combine Sketchy and Intensive logits
                combined_start_logits = (sketchy_start_logits + intensive_start_logits) / 2
                combined_end_logits = (sketchy_end_logits + intensive_end_logits) / 2

                # Answer Verification using embeddings from Intensive Reader
                start_prob = torch.softmax(combined_start_logits, dim=-1)
                end_prob = torch.softmax(combined_end_logits, dim=-1)
                start_indexes = torch.argmax(start_prob, dim=-1)
                end_indexes = torch.argmax(end_prob, dim=-1)

                # Gather the representations of predicted start and end positions
                batch_size = input_ids.size(0)
                start_states = intensive_sequence_output[torch.arange(batch_size), start_indexes]
                end_states = intensive_sequence_output[torch.arange(batch_size), end_indexes]
                verifier_input = torch.cat([start_states, end_states], dim=-1)

                # Answer verification
                verify_logits = self.verifier(verifier_input).squeeze(-1)  # (batch_size,)

                outputs = (combined_start_logits, combined_end_logits, verify_logits)

                if start_positions is not None and end_positions is not None:
                    # Compute loss
                    loss_fct = nn.CrossEntropyLoss()
                    start_loss = loss_fct(combined_start_logits, start_positions)
                    end_loss = loss_fct(combined_end_logits, end_positions)
                    # Verification loss
                    verifier_labels = ((start_positions != 0) & (end_positions != 0)).float()
                    verify_loss_fct = nn.BCELoss()
                    verify_loss = verify_loss_fct(verify_logits, verifier_labels)
                    total_loss = (start_loss + end_loss + verify_loss) / 3
                    outputs = (total_loss,) + outputs

                return outputs  # (loss), start_logits, end_logits, verify_logits

        # Step 4: Initialize the model and training arguments
        model = RetroReader.from_pretrained('bert-base-uncased')

        training_args = TrainingArguments(
            output_dir=args.model_path,
            evaluation_strategy="epoch",
            save_strategy="no",
            learning_rate=3e-5,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
        )

        # Step 5: Implement the compute_metrics function
        metric = evaluate.load("squad")

        def postprocess_qa_predictions(examples, features, predictions, tokenizer, n_best_size=20, max_answer_length=30):
            """
            Post-processes the predictions of a question-answering model to convert them into answers that are substrings of the original contexts.
            """
            all_start_logits, all_end_logits, all_verify_logits = predictions
            example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
            features_per_example = collections.defaultdict(list)
            for i, feature in enumerate(features):
                features_per_example[example_id_to_index[feature["example_id"]]].append(i)

            predictions = collections.OrderedDict()

            for example_index, example in enumerate(examples):
                feature_indices = features_per_example[example_index]
                min_null_score = None
                valid_answers = []

                context = example["context"]

                for feature_index in feature_indices:
                    start_logits = all_start_logits[feature_index]
                    end_logits = all_end_logits[feature_index]
                    offset_mapping = features[feature_index]["offset_mapping"]
                    input_ids = features[feature_index]["input_ids"]

                    # Update minimum null score
                    cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
                    feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                    if min_null_score is None or min_null_score > feature_null_score:
                        min_null_score = feature_null_score

                    # Go through all possible combinations of start and end logits
                    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            # Skip invalid indices
                            if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                                continue
                            if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                                continue
                            # Skip answers that are too long
                            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                                continue
                            # Get the predicted answer text
                            start_char = offset_mapping[start_index][0]
                            end_char = offset_mapping[end_index][1]
                            answer_text = context[start_char:end_char]
                            valid_answers.append(
                                {
                                    "score": start_logits[start_index] + end_logits[end_index],
                                    "text": answer_text,
                                }
                            )

                if valid_answers:
                    best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
                else:
                    # If no valid answers, return empty string
                    best_answer = {"text": "", "score": 0.0}

                predictions[example["id"]] = best_answer["text"]

            return predictions

        def compute_metrics(p: EvalPrediction):
            # Post-process the predictions
            predictions = postprocess_qa_predictions(
                examples=dataset["validation"],
                features=tokenized_datasets["validation"],
                predictions=p.predictions,
                tokenizer=tokenizer,
            )

            # Format predictions and references as required by the metric
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]
            references = [
                {"id": ex["id"], "answers": ex["answers"]}
                for ex in dataset["validation"]
            ]

            # Compute the metric
            result = metric.compute(predictions=formatted_predictions, references=references)
            return result

        # Step 6: Set up the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'].select(range(10000)),
            eval_dataset=tokenized_datasets['validation'].select(range(1000)),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Step 7: Start training
        trainer.train()

        trainer.save_model(args.model_path)

        # Step 8: Evaluate the model
        trainer.evaluate()

    elif args.mode == "inference":
        # Inference mode
        if not args.question or not args.context:
            print("Please provide both a question and a context for inference.")
            return

        # Load the trained model
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model_path)

        # Define the RetroReader model class (include the full class definition)
        class RetroReader(BertPreTrainedModel):
            def __init__(self, config):
                super(RetroReader, self).__init__(config)
                self.config = config

                # Sketchy Reader Encoder (shared with Intensive Reader for efficiency)
                self.bert = BertModel(config)
                
                # Sketchy Reader Output Layer
                self.sketchy_qas_output = nn.Linear(config.hidden_size, 2)

                # Intensive Reader Encoder (additional layer)
                self.intensive_layer = nn.Linear(config.hidden_size, config.hidden_size)

                # Intensive Reader Output Layer
                self.intensive_qas_output = nn.Linear(config.hidden_size, 2)

                # Answer Verification Module
                self.verifier = nn.Sequential(
                    nn.Linear(2 * config.hidden_size, config.hidden_size),
                    nn.Tanh(),
                    nn.Linear(config.hidden_size, 1),
                    nn.Sigmoid()
                )

                self.init_weights()

            def forward(
                self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                start_positions=None,
                end_positions=None,
            ):
                # Sketchy Reading Stage
                outputs = self.bert(
                    input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
                )  # (batch_size, seq_len, hidden_size)
                sequence_output = outputs[0]

                # Sketchy Reader Start and End logits
                sketchy_logits = self.sketchy_qas_output(sequence_output)
                sketchy_start_logits, sketchy_end_logits = sketchy_logits.split(1, dim=-1)
                sketchy_start_logits = sketchy_start_logits.squeeze(-1)
                sketchy_end_logits = sketchy_end_logits.squeeze(-1)

                # Get top k start and end positions from Sketchy Reader
                k = 5  # Number of top positions to consider
                sketchy_start_prob = torch.softmax(sketchy_start_logits, dim=-1)
                sketchy_end_prob = torch.softmax(sketchy_end_logits, dim=-1)

                # Intensive Reading Stage
                intensive_sequence_output = self.intensive_layer(sequence_output)  # (batch_size, seq_len, hidden_size)

                # Intensive Reader Start and End logits
                intensive_logits = self.intensive_qas_output(intensive_sequence_output)
                intensive_start_logits, intensive_end_logits = intensive_logits.split(1, dim=-1)
                intensive_start_logits = intensive_start_logits.squeeze(-1)
                intensive_end_logits = intensive_end_logits.squeeze(-1)

                # Combine Sketchy and Intensive logits
                combined_start_logits = (sketchy_start_logits + intensive_start_logits) / 2
                combined_end_logits = (sketchy_end_logits + intensive_end_logits) / 2

                # Answer Verification using embeddings from Intensive Reader
                start_prob = torch.softmax(combined_start_logits, dim=-1)
                end_prob = torch.softmax(combined_end_logits, dim=-1)
                start_indexes = torch.argmax(start_prob, dim=-1)
                end_indexes = torch.argmax(end_prob, dim=-1)

                # Gather the representations of predicted start and end positions
                batch_size = input_ids.size(0)
                start_states = intensive_sequence_output[torch.arange(batch_size), start_indexes]
                end_states = intensive_sequence_output[torch.arange(batch_size), end_indexes]
                verifier_input = torch.cat([start_states, end_states], dim=-1)

                # Answer verification
                verify_logits = self.verifier(verifier_input).squeeze(-1)  # (batch_size,)

                outputs = (combined_start_logits, combined_end_logits, verify_logits)

                return outputs  # start_logits, end_logits, verify_logits

        # Load the model
        model = RetroReader.from_pretrained(args.model_path, config=config)
        model.eval()  # Set model to evaluation mode

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Prepare inputs
        question = args.question
        context = args.context

        # Tokenize input
        inputs = tokenizer(
            question,
            context,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Keep the offset mapping for post-processing
        offset_mapping = inputs.pop("offset_mapping")

        # Run the model
        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs[0]  # Start logits
        end_logits = outputs[1]    # End logits
        verify_logits = outputs[2] # Verification logits

        # Convert logits to probabilities
        start_probs = torch.softmax(start_logits, dim=-1).cpu().numpy()
        end_probs = torch.softmax(end_logits, dim=-1).cpu().numpy()
        verify_probs = verify_logits.cpu().numpy()

        # Get the most probable start and end indices
        start_index = np.argmax(start_probs, axis=1)[0]
        end_index = np.argmax(end_probs, axis=1)[0]

        # Ensure that the end index is greater than or equal to the start index
        if end_index < start_index:
            end_index = start_index

        # Use the offset mapping to get the character positions
        offsets = offset_mapping.cpu().numpy()[0]

        start_char = offsets[start_index][0]
        end_char = offsets[end_index][1]

        # Extract the answer from the context
        predicted_answer = context[start_char:end_char]

        # Answer verification
        verification_threshold = 0.5  # Adjust based on your requirements
        is_answer_valid = verify_probs[0] > verification_threshold
        print(verify_probs[0])
        print("\nQuestion:", question)
        if is_answer_valid:
            print("Predicted Answer:", predicted_answer)
        else:
            print("The model is not confident about the answer.")

    else:
        print("Invalid mode selected. Please choose 'train' or 'inference'.")

if __name__ == "__main__":
    main()
