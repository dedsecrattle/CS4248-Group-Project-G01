import torch
from transformers import BertTokenizerFast, AutoConfig
import torch.nn as nn
import numpy as np

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load the trained model
model_path = './results/checkpoint-1668'  # Replace with your actual model path if different
config = AutoConfig.from_pretrained(model_path)

# Define the RetroReader model class (include the full class definition)
from transformers import BertPreTrainedModel, BertModel

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

        # Get the most probable start and end positions
        start_prob = torch.softmax(start_logits, dim=-1)
        end_prob = torch.softmax(end_logits, dim=-1)
        start_indexes = torch.argmax(start_prob, dim=-1)
        end_indexes = torch.argmax(end_prob, dim=-1)

        # Gather the representations of predicted start and end positions
        batch_size = input_ids.size(0)
        start_states = sequence_output[torch.arange(batch_size), start_indexes]
        end_states = sequence_output[torch.arange(batch_size), end_indexes]
        verifier_input = torch.cat([start_states, end_states], dim=-1)

        # Answer verification
        verify_logits = self.verifier(verifier_input).squeeze(-1)  # (batch_size,)

        outputs = (start_logits, end_logits, verify_logits)
        
        if start_positions is not None and end_positions is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # Verification loss
            verifier_labels = ((start_positions != 0) & (end_positions != 0)).float()
            verify_loss_fct = nn.BCELoss()
            verify_loss = verify_loss_fct(verify_logits, verifier_labels)
            total_loss = (start_loss + end_loss + verify_loss) / 3
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, verify_logits

# Load the model
model = RetroReader.from_pretrained(model_path, config=config)
model.eval()  # Set model to evaluation mode

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Example question and context
question = "Who wrote the novel '1984'?"
context = """
'1985' is a Novel written by Prabhat and
'1984' is a dystopian social science fiction novel and cautionary tale by English writer George Orwell.
It was published on 8 June 1949 by Secker & Warburg as Orwell's ninth and final book completed in his lifetime.
"""

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
verification_threshold = 0.95  # Adjust based on your requirements
print(verify_probs[0])
is_answer_valid = verify_probs[0] > verification_threshold

print("Question:", question)
if is_answer_valid:
    print("Predicted Answer:", predicted_answer)
else:
    print("The model is not confident about the answer.")
