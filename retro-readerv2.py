import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
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

class RetroReader(nn.Module):
    
    def __init__(
        self,
        model_name="bert-base-uncased",  
        weight_sketchy=0.5,
        weight_intensive=0.5,
        is_squad_v2=False,
    ):
        super(RetroReader, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.is_squad_v2 = is_squad_v2
        
        if self.is_squad_v2:
            self.answerable_head = nn.Linear(
                self.encoder.config.hidden_size, 2
            )
        
        self.cross_attention = nn.MultiheadAttention(
            self.encoder.config.hidden_size, num_heads=8
        )
        
        self.start_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.end_head = nn.Linear(self.encoder.config.hidden_size, 1)
        
        self.verifier_head = nn.Linear(self.encoder.config.hidden_size, 1)
        
        self.weight_sketchy = weight_sketchy
        self.weight_intensive = weight_intensive

    def forward(self, input_ids, attention_mask, question_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        cls_token = sequence_output[:, 0, :]
        
        if self.is_squad_v2:
            answer_logits = self.answerable_head(cls_token)
            score_null = answer_logits[:, 1] - answer_logits[:, 0]
        else:
            score_null = 0

        question_tokens = sequence_output * question_mask.unsqueeze(-1)
        
        question_attended, _ = self.cross_attention(
            question_tokens.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2),
            sequence_output.permute(1, 0, 2),
        )
        question_attended = question_attended.permute(1, 0, 2)

        start_logits = self.start_head(question_attended).squeeze(-1)
        end_logits = self.end_head(question_attended).squeeze(-1)

        verifier_score = self.verifier_head(cls_token)

        score_has = start_logits.max(dim=1).values + end_logits.max(dim=1).values

        final_score = (
            self.weight_sketchy * score_null 
            + self.weight_intensive * score_has
            + verifier_score.squeeze(-1)
        )

        if self.is_squad_v2:
            final_decision = final_score > 0
            return final_decision, start_logits, end_logits, answer_logits, verifier_score
        else:
            final_decision = torch.ones_like(final_score, dtype=torch.bool)
            return final_decision, start_logits, end_logits, verifier_score


def prepare_training_data_squad_v1(tokenizer, max_length=384):
    dataset = load_dataset("squad")

    def preprocess(examples):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=max_length,
            truncation="only_second",
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

                token_start_index = 0
                while (
                    token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                token_start_index -= 1

                token_end_index = len(offsets) - 1
                while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                token_end_index += 1

                if (
                    offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char
                ):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    start_positions.append(token_start_index)
                    end_positions.append(token_end_index)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = dataset.map(
        preprocess, 
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset


def prepare_training_data_squad_v2(tokenizer, max_length=384):
    dataset = load_dataset("squad_v2")

    def preprocess(examples):
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        start_positions = []
        end_positions = []
        answerable_labels = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = inputs["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["text"]) == 0 or len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                answerable_labels.append(0)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                sequence_ids = inputs.sequence_ids(i)

                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                    answerable_labels.append(0)
                else:
                    start_idx = context_start
                    while start_idx <= context_end and offsets[start_idx][0] <= start_char:
                        start_idx += 1
                    start_positions.append(start_idx - 1)

                    end_idx = context_end
                    while end_idx >= context_start and offsets[end_idx][1] >= end_char:
                        end_idx -= 1
                    end_positions.append(end_idx + 1)

                    answerable_labels.append(1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["answerable_labels"] = answerable_labels
        return inputs

    tokenized_dataset = dataset.map(
        preprocess, batched=True, remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset


def prepare_prediction_data(dataset, tokenizer, max_length=384):
    def preprocess(examples):
      
        tokenized_inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
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
        remove_columns=dataset.column_names
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

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
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
            question_mask = (
                (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1
            )

            optimizer.zero_grad()
            if is_squad_v2:
                (
                    final_decision,
                    start_logits,
                    end_logits,
                    answer_logits,
                    verifier_score,
                ) = model(input_ids, attention_mask, question_mask)
                answerable_labels = batch["answerable_labels"].to(device)
                answerable_loss = nn.CrossEntropyLoss()(answer_logits, answerable_labels)
            else:
                (
                    final_decision,
                    start_logits,
                    end_logits,
                    verifier_score,
                ) = model(input_ids, attention_mask, question_mask)

            start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
            end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)
            if is_squad_v2:
                loss = start_loss + end_loss + answerable_loss
            else:
                loss = start_loss + end_loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

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
                question_mask = (
                    (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1
                )

                if is_squad_v2:
                    (
                        final_decision,
                        start_logits,
                        end_logits,
                        answer_logits,
                        verifier_score,
                    ) = model(input_ids, attention_mask, question_mask)
                    answerable_labels = batch["answerable_labels"].to(device)
                    answerable_loss = nn.CrossEntropyLoss()(answer_logits, answerable_labels)
                else:
                    (
                        final_decision,
                        start_logits,
                        end_logits,
                        verifier_score,
                    ) = model(input_ids, attention_mask, question_mask)

                start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
                end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)
                if is_squad_v2:
                    val_loss = start_loss + end_loss + answerable_loss
                else:
                    val_loss = start_loss + end_loss
                total_val_loss += val_loss.item()

                val_bar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

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
):
    data_collator = CustomDataCollator(tokenizer)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = {}

    for batch in tqdm(val_loader, desc="Generating predictions"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        question_mask = (
            (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1
        )
        original_ids = batch["original_id"]

        with torch.no_grad():
            if is_squad_v2:
                (
                    final_decision,
                    start_logits,
                    end_logits,
                    answer_logits,
                    verifier_score,
                ) = model(input_ids, attention_mask, question_mask)
            else:
                (
                    final_decision,
                    start_logits,
                    end_logits,
                    verifier_score,
                ) = model(input_ids, attention_mask, question_mask)

        for i in range(input_ids.size(0)):
            if is_squad_v2:
                answerable_prob = torch.softmax(answer_logits[i], dim=-1)
                is_answerable = answerable_prob.argmax() == 1
                if is_answerable:
                    start_idx = torch.argmax(start_logits[i]).item()
                    end_idx = torch.argmax(end_logits[i]).item()
                    if start_idx > end_idx:
                        start_idx, end_idx = end_idx, start_idx
                    answer = tokenizer.decode(
                        input_ids[i][start_idx : end_idx + 1], skip_special_tokens=True
                    ).strip()
                else:
                    answer = ""
            else:
                start_idx = torch.argmax(start_logits[i]).item()
                end_idx = torch.argmax(end_logits[i]).item()
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                answer = tokenizer.decode(
                    input_ids[i][start_idx : end_idx + 1], skip_special_tokens=True
                ).strip()

            question_id = original_ids[i]
            if question_id not in predictions or len(predictions[question_id]) == 0:
                predictions[question_id] = answer

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Predictions saved to {output_file}")
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate the RetroReader model on SQuAD dataset."
    )

    parser.add_argument(
        "--train", action="store_true", help="Whether to train the model."
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
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name.",
    )
    parser.add_argument(
        "--max_length", type=int, default=384, help="Maximum sequence length."
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

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(args.dataset)

    is_squad_v2 = args.dataset == "squad_v2"

    if args.train:
        print("Preparing training data...")
        if is_squad_v2:
            tokenized_dataset = prepare_training_data_squad_v2(
                tokenizer, max_length=args.max_length
            )
        else:
            tokenized_dataset = prepare_training_data_squad_v1(
                tokenizer, max_length=args.max_length
            )

        train_dataset = tokenized_dataset["train"]
        validation_dataset = tokenized_dataset["validation"]

        model = RetroReader(model_name=args.model_name, is_squad_v2=is_squad_v2)
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
    else:
        print("Preparing prediction data...")
        val_dataset = prepare_prediction_data(
            dataset["validation"], tokenizer, max_length=args.max_length
        )
        model = RetroReader(model_name=args.model_name, is_squad_v2=is_squad_v2)
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