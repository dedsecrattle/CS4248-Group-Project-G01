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

dataset = load_dataset('squad')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    questions = [q.lstrip() for q in examples['question']]
    contexts = examples['context']
    answers = examples['answers']
    
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

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    inputs["start_positions"] = []
    inputs["end_positions"] = []
    inputs["offset_mapping"] = offset_mapping  # For evaluation

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]
        if len(answer["answer_start"]) == 0:
            inputs["start_positions"].append(0)
            inputs["end_positions"].append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
                inputs["start_positions"].append(0)
                inputs["end_positions"].append(0)
            else:
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

tokenized_datasets = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names
)

# Step 4: Define the Retro-Reader model
class RetroReader(BertPreTrainedModel):
    def __init__(self, config):
        super(RetroReader, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # For start and end logits
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
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)

        start_prob = torch.softmax(start_logits, dim=-1)
        end_prob = torch.softmax(end_logits, dim=-1)
        start_indexes = torch.argmax(start_prob, dim=-1)
        end_indexes = torch.argmax(end_prob, dim=-1)

        batch_size = input_ids.size(0)
        start_states = sequence_output[torch.arange(batch_size), start_indexes]
        end_states = sequence_output[torch.arange(batch_size), end_indexes]
        verifier_input = torch.cat([start_states, end_states], dim=-1)


        verify_logits = self.verifier(verifier_input).squeeze(-1)  # (batch_size,)

        outputs = (start_logits, end_logits, verify_logits)
        
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            verifier_labels = ((start_positions != 0) & (end_positions != 0)).float()
            verify_loss_fct = nn.BCELoss()
            verify_loss = verify_loss_fct(verify_logits, verifier_labels)
            total_loss = (start_loss + end_loss + verify_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs

model = RetroReader.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

from transformers.trainer_utils import EvalPrediction

metric = evaluate.load("squad")

def postprocess_qa_predictions(examples, features, predictions, tokenizer, n_best_size=20, max_answer_length=30):
    """
    Post-processes the predictions of a question-answering model to convert them into answers that are substrings of the original contexts.

    Args:
        examples: The non-tokenized dataset.
        features: The tokenized dataset.
        predictions: Tuple containing start_logits and end_logits.
        tokenizer: The tokenizer used for encoding the data.
        n_best_size: The total number of n-best predictions to generate.
        max_answer_length: The maximum length of an answer that can be generated.

    Returns:
        A dictionary mapping example IDs to predicted answer texts.
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

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score > feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
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
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions

def compute_metrics(p: EvalPrediction):
    predictions = postprocess_qa_predictions(
        examples=dataset["validation"],
        features=tokenized_datasets["validation"],
        predictions=p.predictions,
        tokenizer=tokenizer,
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    references = [
        {"id": ex["id"], "answers": ex["answers"]}
        for ex in dataset["validation"]
    ]

    result = metric.compute(predictions=formatted_predictions, references=references)
    return result

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].select(range(10000)),
    eval_dataset=tokenized_datasets['validation'].select(range(1000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()
