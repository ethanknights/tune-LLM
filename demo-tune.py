# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import logging

logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct")



def generate_dataset():
    # Define your dataset as a list of tuples, where each tuple contains the user query and its corresponding issue category

    dataset = [
        ("There's a large pothole on Main Street.", "Pothole"),
        ("I hit a pothole on Elm Avenue and my tire popped.", "Pothole"),
        ("There's a deep pothole near the intersection of Oak and Pine streets.", "Pothole"),
        ("There's graffiti on the park wall.", "Graffiti"),
        ("Someone spray-painted graffiti on the bus stop.", "Graffiti"),
        ("There's offensive graffiti on the side of the building downtown.", "Graffiti"),
        ("There's loud construction noise coming from the building next door.", "Noise Complaint"),
        ("My neighbor's party is too loud and it's keeping me awake.", "Noise Complaint"),
        ("The neighbors are playing loud music late at night.", "Noise Complaint"),
        ("The trash bins haven't been emptied for days.", "Trash Pickup Request"),
        ("There's trash scattered all over the sidewalk.", "Trash Pickup Request"),
        ("The garbage truck missed our street on pickup day.", "Trash Pickup Request"),
        ("The street light at the corner of Maple and Elm streets is out.", "Street Light Outage"),
        ("There's a dark area on the street because the light isn't working.", "Street Light Outage"),
        ("The street light flickers on and off intermittently.", "Street Light Outage"),
    ]

    # Shuffle the dataset to ensure randomness during training
    import random
    random.shuffle(dataset)

    # Split the dataset into training and validation sets (you can adjust the split ratio as needed)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    valid_dataset = dataset[train_size:]

    # Print the first few examples in the training set for verification
    print("Training examples:")
    for example in train_dataset[:5]:
        print(example)

    # Print the first few examples in the validation set for verification
    print("\nValidation examples:")
    for example in valid_dataset[:5]:
        print(example)

    return train_dataset, valid_dataset


train_texts, valid_texts = generate_dataset()



# Step 3: Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True)

# Step 4: Create TextDataset and DataCollator
train_dataset = TextDataset(train_encodings, tokenizer=tokenizer)
valid_dataset = TextDataset(valid_encodings, tokenizer=tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 5: Define TrainingArguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Save the fine-tuned model
model.save_pretrained("fine_tuned_llama3_model")
