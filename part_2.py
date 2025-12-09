# https://huggingface.co/docs/transformers/en/model_doc/distilbert 
https://huggingface.co/docs/transformers/en/model_doc/distilbert#transformers.DistilBertForMaskedLM.forward.example
#https://huggingface.co/docs/transformers/en/trainer


from datasets import load_dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments
import evaluate
import numpy as np
from transformers import Trainer

model_name = "distilbert-base-uncased"
max_len = 128

def tokenize_data(max_len):
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    def tokenize_fn(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    encoded_dataset = dataset.map(tokenize_fn, batched=True)
    train_data = encoded_dataset["train"]
    val_data = encoded_dataset["validation"]
    return tokenizer, train_data, val_data

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_metric = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_metric = f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy_metric, "f1": f1_metric}

tokenizer, train_data, val_data = tokenize_data(max_len=max_len)

model = DistilBertForSequenceClassification.from_pretrained(model_name=model_name, 
                                                            num_labels=2)

training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_evaluation="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model_name,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate()
print(results)

trained_model = "distilbert_sst2"
trainer.save_model(trained_model)
tokenizer.save_pretrained(trained_model)

# Output: [{'label': 'POSITIVE', 'score': 0.9998}]