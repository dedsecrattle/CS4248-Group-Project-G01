import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import LayerNorm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from tqdm import tqdm
import json
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RetroReader(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-base",
        weight_sketchy=0.5,
        weight_intensive=0.5,
        is_squad_v2=False,
    ):
        super(RetroReader, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.is_squad_v2 = is_squad_v2
        dropout_rate = 0.2
        if self.is_squad_v2:
            self.answerable_head = nn.Linear(self.encoder.config.hidden_size, 2)
            self.verification_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.start_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )
        self.end_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(0.3)
        self.weight_sketchy = weight_sketchy
        self.weight_intensive = weight_intensive

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        cls_token = self.dropout(sequence_output[:, 0, :])

        start_logits = self.start_head(sequence_output).squeeze(-1)
        end_logits = self.end_head(sequence_output).squeeze(-1)

        if self.is_squad_v2:
            answer_logits = self.answerable_head(cls_token)
            start_pred = torch.argmax(start_logits, dim=-1)
            end_pred = torch.argmax(end_logits, dim=-1)
            start_output = sequence_output[range(len(start_pred)), start_pred]
            end_output = sequence_output[range(len(end_pred)), end_pred]
            verifier_input = (start_output + end_output) / 2
            verifier_logits = self.verification_head(verifier_input).squeeze(-1)
        else:
            answer_logits = None


        if self.is_squad_v2:
            return start_logits, end_logits, answer_logits, verifier_logits
        else:
            return start_logits, end_logits


def prepare_training_data_squad_v1(tokenizer, max_length=384):
    dataset = load_dataset("squad")

    def preprocess(examples):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_mapping = inputs.pop("overflow_to_sample_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = inputs["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                sequence_ids = inputs.sequence_ids(i)

                idx = 0
                while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    token_start_index = context_start
                    while token_start_index <= context_end and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    token_end_index = context_end
                    while token_end_index >= context_start and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset


def prepare_training_data_squad_v2(tokenizer, max_length=384):
    dataset = load_dataset("squad_v2")

    def preprocess(examples):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            stride=128,
            truncation=True,
            max_length=512,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        start_positions = []
        end_positions = []
        answerable_labels = []
        verification_labels = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = inputs["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["text"]) == 0 or len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                answerable_labels.append(0)
                verification_labels.append(0)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                sequence_ids = inputs.sequence_ids(i)

                idx = 0
                while idx < len(sequence_ids) and sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                    answerable_labels.append(0)
                    verification_labels.append(0)
                else:
                    token_start_index = context_start
                    while token_start_index <= context_end and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    token_end_index = context_end
                    while token_end_index >= context_start and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)
                    
                    answerable_labels.append(1)
                    verification_labels.append(1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["answerable_labels"] = answerable_labels
        inputs["verification_labels"] = verification_labels
        return inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset


def prepare_prediction_data(dataset, tokenizer, max_length=384):
    def preprocess(examples):
        tokenized_inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
        )


        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")

        original_ids = [
            str(examples["id"][sample_mapping[i]])
            for i in range(len(tokenized_inputs["input_ids"]))
        ]

        tokenized_inputs["original_id"] = original_ids

        return tokenized_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        original_ids = [feature.pop("original_id") for feature in features]
        batch = self.data_collator(features)
        batch["original_id"] = original_ids
        return batch


def train_model_and_save(
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    is_squad_v2=False,
    epochs=3,
    batch_size=8,
    lr=3e-5,
    save_path="retro_reader_model.pth",
):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_training_steps = epochs * len(train_loader)
    warmup_steps = int(0.2 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            optimizer.zero_grad()


            if is_squad_v2:
                start_logits, end_logits, answer_logits, verifier_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask)
                answerable_labels = batch["answerable_labels"].to(device)
                verifier_labels = batch["verification_labels"].to(device)

                answerable_loss = nn.CrossEntropyLoss()(answer_logits, answerable_labels)
                verifier_loss = nn.BCEWithLogitsLoss()(verifier_logits, verifier_labels.float())
            else:
                start_logits, end_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                answerable_loss = 0
                verifier_loss = 0

            start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
            end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)

            loss = start_loss + end_loss + answerable_loss + verifier_loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}")

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                verification_loss = 0
                answerable_loss = 0
                if is_squad_v2:
                    start_logits, end_logits, answer_logits, verifier_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    answerable_labels = batch["answerable_labels"].to(device)
                    verification_labels = batch["verification_labels"].to(device)
                    answerable_loss = nn.CrossEntropyLoss()(answer_logits, answerable_labels)
                    verification_loss = nn.BCEWithLogitsLoss()(verifier_logits, verification_labels.float())
                else:
                    start_logits, end_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
                end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)

                val_loss = start_loss + end_loss + answerable_loss + verification_loss
                total_val_loss += val_loss.item()

                val_bar.set_postfix({"loss": f"{val_loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model_and_predict(
    model,
    val_dataset,
    tokenizer,
    is_squad_v2=False,
    model_path="retro_reader_model.pth",
    output_file="predictions.json",
    batch_size=8,
    verifier_threshold=0.5
):
    data_collator = CustomDataCollator(tokenizer)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    predictions = {}

    for batch in tqdm(val_loader, desc="Generating predictions"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        original_ids = batch["original_id"]

        with torch.no_grad():
            if is_squad_v2:
                start_logits, end_logits, answer_logits, verifier_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                answerable_probs = torch.softmax(answer_logits, dim=-1)
                is_answerable = answerable_probs.argmax(dim=-1) == 1
            else:
                start_logits, end_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                is_answerable = torch.ones(input_ids.size(0), dtype=torch.bool).to(device)

        start_positions = torch.argmax(start_logits, dim=-1)
        end_positions = torch.argmax(end_logits, dim=-1)

        for i in range(input_ids.size(0)):
            verifier_score = (
                torch.sigmoid(verifier_logits[i]).item() if is_squad_v2 else 1.0
            )
            if not is_answerable[i] or (is_squad_v2 and verifier_score < verifier_threshold):
                answer = ""
            else:
                start_idx = start_positions[i].item()
                end_idx = end_positions[i].item()

                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx

                answer = tokenizer.decode(
                    input_ids[i][start_idx : end_idx + 1],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()

            question_id = original_ids[i]
            if question_id not in predictions:
                predictions[question_id] = answer

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to {output_file}")
    return predictions


def inference_with_question_context(
    model,
    tokenizer,
    question,
    context,
    is_squad_v2=False,
    model_path="retro_reader_model.pth",
    verifier_threshold=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=512,
        stride=128,
        return_tensors="pt",
        padding="max_length",
        return_offsets_mapping=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        if is_squad_v2:
            start_logits, end_logits, answer_logits, verifier_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            answerable_probs = torch.softmax(answer_logits, dim=-1)
            is_answerable = answerable_probs.argmax(dim=-1) == 1
        else:
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            is_answerable = torch.ones(input_ids.size(0), dtype=torch.bool).to(device)

        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()

        if is_squad_v2:
            verifier_score = torch.sigmoid(verifier_logits).item()
            if not is_answerable or verifier_score < verifier_threshold:
                return "The answer is not available in context"

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        answer = tokenizer.decode(
            input_ids[0][start_idx: end_idx + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        
        return answer

def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the RetroReader model on SQuAD dataset."
    )

    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model."
    )
    parser.add_argument(
        "--inference", action="store_true", help="Run inference with a given question and context."
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument(
        "--save_path",
        type=str,
        default="retro_reader_model.pth",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="retro_reader_model.pth",
        help="Path to load the trained model.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="File to save predictions.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad_v2",
        choices=["squad", "squad_v2"],
        help="Dataset to use: 'squad' for SQuAD v1.1, 'squad_v2' for SQuAD v2.0",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question for inference mode."
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Context for inference mode."
    )

    args = parser.parse_args()

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    is_squad_v2 = args.dataset == "squad_v2"

    if args.train:
        print("Preparing training data...")
        if is_squad_v2:
            tokenized_dataset = prepare_training_data_squad_v2(
                tokenizer
            )
        else:
            tokenized_dataset = prepare_training_data_squad_v1(
                tokenizer
            )

        train_dataset = tokenized_dataset["train"]
        validation_dataset = tokenized_dataset["validation"]

        model = RetroReader(model_name="microsoft/deberta-base", is_squad_v2=is_squad_v2)
        print("Starting training...")
        train_model_and_save(
            model=model,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            tokenizer=tokenizer,
            is_squad_v2=is_squad_v2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.save_path,
        )

    elif args.inference:
        if not args.question or not args.context:
            print("Please provide both --question and --context for inference mode.")
            return
        model = RetroReader(model_name="microsoft/deberta-base", is_squad_v2=is_squad_v2)
        answer = inference_with_question_context(
            model=model,
            tokenizer=tokenizer,
            question=args.question,
            context=args.context,
            is_squad_v2=is_squad_v2,
            model_path=args.model_path,
        )

        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
    else:
        print("Preparing prediction data...")
        dataset = load_dataset(args.dataset)
        val_dataset = prepare_prediction_data(
            dataset["validation"], tokenizer)
        model = RetroReader(model_name="microsoft/deberta-base", is_squad_v2=is_squad_v2)
        print("Loading model and generating predictions...")
        predictions = load_model_and_predict(
            model=model,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            is_squad_v2=is_squad_v2,
            model_path=args.model_path,
            output_file=args.output_file,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()


