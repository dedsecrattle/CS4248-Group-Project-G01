# CS4248-Group-Project-G01

This project aims to develop an efficient Extractive QA system using the Retrospective Reader model, focusing on improving accuracy and real-time performance for tasks like search engines and virtual assistants. It will be evaluated on benchmark datasets like SQuAD 2.0 to ensure effectiveness with both answerable and unanswerable questions.

To test the Model First Train it :-

`python retro_reader.py --mode train --model_path ./model"`

Now you can make Inference using the following Command :-

`python retro_reader.py --mode inference --model_path ./model --question "Who wrote the novel '1984'?" --context "'1984' is a dystopian social science fiction novel and cautionary tale written by English writer George Orwell. It was published on 8 June 1949 by Secker & Warburg as Orwell's ninth and final book completed in his lifetime."`
