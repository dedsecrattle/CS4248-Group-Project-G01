Here’s the complete markdown code for your project documentation, including the properly formatted code blocks for users to easily copy commands:

# CS4248-Group-Project-G01

This project aims to develop an efficient Extractive QA system using the Retrospective Reader model, focusing on improving accuracy and real-time performance for tasks like search engines and virtual assistants. It will be evaluated on benchmark datasets like SQuAD 2.0 to ensure effectiveness with both answerable and unanswerable questions.

## Training the Model

To train the model, use the following command:

```bash
python modelv1.py --mode train --model_path ./model
```

## Making an Inference

After training, you can make an inference with the following command:

```bash
python modelv1.py --mode inference --model_path ./model --question "Who wrote the novel '1984'?" --context "'1984' is a dystopian social science fiction novel and cautionary tale written by English writer George Orwell. It was published on 8 June 1949 by Secker & Warburg as Orwell's ninth and final book completed in his lifetime."
```

Make sure to replace the model path and other parameters as needed based on your setup.

Due to Compute Limiations currently the model will be trained only on 10,000 data from Squad V1.1 , but you change it in the `modelv1.py`