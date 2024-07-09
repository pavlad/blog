# Named Entity Recognition with BERT

As a follow-up to my previous endeavor of writing transformer architecture from scratch, I decided to learn more about the application of BERT and how it can be used for Named Entity Recognition. For this, I'm following the outline of this [HuggingFace tutorial](https://huggingface.co/learn/nlp-course/chapter7/2?fw=tf), but I'll be using a different dataset and will dig a bit deeper into certain concepts.

# 1. Intro

## 1.1 What is Named Entity Recognition?

<div align="center">
<img src="https://cdn.botpenguin.com/assets/website/Named_Entity_Recognition_NER_ca897448c8.png" alt="drawing" width="400"/><br>
Source: botpenguin.com
</div><br>


NER is a use of Natural Language Processing to categorize entities like people, locations, and organizations from unstructured text data. This technique can be applied in many organizations dealing with unstructured data.

## 1.2 What is BERT?
<div align="center">
<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs10462-021-09958-2/MediaObjects/10462_2021_9958_Fig6_HTML.png" alt="drawing" width="400"/><br>
Source: Henry Nunoo-Mensah
</div><br>


BERT stands for Bidirectional Encoder Representations from Transformers, which is an architecture based on the encoder part of the transformer architecture. It is often used for many kinds of language-related tasks, including sentiment analysis.

In the past, neural network architectures like RNNs and LSTMs were often used for language-related tasks, but BERT offers a few advantages:
- It uses an attention mechanism, allowing it to train much faster than RNNs. More advantages are listed here.
- As the name states, during pre-training, the model is bidirectional, which allows it to understand context from both past and future tokens at the same time, allowing it to understand the context much better.

## 1.3 Pre-training BERT
What makes BERT so useful is its great understanding of language, which is obtained by pre-training the model on a large amount of data. This consists of two parts that happen simultaneously:

<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1270/1*i8zICfESnaGt4EVRcWBLKw.png" alt="drawing" width="400"/><br>
Source: Renu Khanderwal
</div><br>


- **Masked Language Modeling**: sequences of embedded tokens are fed to the model, and some words are randomly masked. The model is trained to predict what the missing tokens are, allowing it to learn about the relationships between words or partial words.
- **Next Sentence Prediction**: the model is given two sentences A and B, and the model is trained to predict a binary classification: whether sentence B comes after A. This trains the model on larger sentence structures.

## 1.4 Objective

My objective is to finetune a Dutch BERT model, [bert-base-dutch-cased](https://huggingface.co/GroNLP/bert-base-dutch-cased), on a named entity recognition task.

We can benefit from transfer learning by using an already existing BERT model, [bert-base-dutch-cased](https://huggingface.co/GroNLP/bert-base-dutch-cased), pre-trained by the University of Groningen. We can then attach a classification layer for the model to predict named entities, for which I'll use the [unimelb-nlp/wikiann](https://huggingface.co/datasets/unimelb-nlp/wikiann) dataset, which contains Dutch NER labels taken from Wikipedia. The dataset includes 20k training, 10k validation, and 10k test examples.

Why Dutch? I speak Dutch, and I was interested in seeing how well the pretrained BERT models work.


## 2. Dataset

Let's start off with loading in the dataset.


```python
!pip install datasets

```

```python
from datasets import load_dataset

raw_datasets = load_dataset('unimelb-nlp/wikiann', 'nl', trust_remote_code=True)
```

We can see that the dataset exists out of `tokens` and `ner_tag` values.


```python
raw_datasets['train'][0]
```

    {'tokens': ['Het',
      'zou',
      'zijn',
      'enige',
      'optreden',
      'in',
      'de',
      'hoofdmacht',
      'van',
      'FC',
      'Twente',
      'worden',
      '.'],
     'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0],
     'langs': ['nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl',
      'nl'],
     'spans': ['ORG: FC Twente']}



Let's take a look at the labels.


```python
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names
label_names
```




    ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']




```python
words = raw_datasets["train"][4]["tokens"]
labels = raw_datasets["train"][4]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)
```

    Ook Macrobius , Paul  Henri Thiry d'Holbach en Stephen Hawking hebben zich over het probleem uitgesproken . 
    O   B-PER     O B-PER I-PER I-PER I-PER     O  B-PER   I-PER   O      O    O    O   O        O            O 


We can see that the NER tags have four entities:
- `O` is used for tokens that do not belong to any entity
- `PER` is used for persons
- `ORG` is used for organisations
- `LOG` is used for locations

And there are two prefixes:
- `B` incicates the beginning of an entity
- `I` indicates the inside of an entity

What this basically means is that when a tokenizer is split into a subword, these prefixes indicate that a sequence of subwords is part of the same entity.

## 3. Tokenization
As we now have the input sequences with their corresponding NER tags, we can continue with the tokenization of the data.

We'll use the [GroNLP/bert-base-dutch-cased](https://huggingface.co/GroNLP/bert-base-dutch-cased) tokenizer which is case-sensitive. The case sensitivity is rather important here, as often brands are capitalized, and so are acronyms. This will allow the model to more accurately identify entities.


```python
from transformers import AutoTokenizer

model_name = 'GroNLP/bert-base-dutch-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name) # HuggingFace helper function which provides a unified interface
```


```python
inputs = tokenizer(raw_datasets['train'][0]['tokens'], is_split_into_words=True)
inputs.tokens()
```




    ['[CLS]',
     'Het',
     'zou',
     'zijn',
     'enige',
     'optreden',
     'in',
     'de',
     'hoofd',
     '##macht',
     'van',
     'FC',
     'Twente',
     'worden',
     '.',
     '[SEP]']



As out tokenizer splits words into subwords, we have to deal with the mismatch between our tokenized inputs and the labels. We can solve this issue by creating a function that aligns labels with their corresponding (partial) tokens. Special tokens get a label of `-100` as per conventions.


```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        label = -100
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
        new_labels.append(label)
    return new_labels
```

Let's test the function out.


```python
labels = raw_datasets['train'][0]['ner_tags']
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0]
    [-100, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 3, 4, 0, 0, -100]


We can now make a function that will allow us to tokenize the input and add the new labels for the whole dataset.


```python
def tokenize_and_align_labels(data):
  tokenized_inputs = tokenizer(
      data['tokens'],
      truncation=True,
      is_split_into_words=True
  )
  all_labels = data['ner_tags']
  new_labels = []

  for i, labels in enumerate(all_labels):
    word_ids = tokenized_inputs.word_ids(i)
    new_labels.append(align_labels_with_tokens(labels, word_ids))

  tokenized_inputs['labels'] = new_labels
  return tokenized_inputs

```


```python
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets['train'].column_names
)
```


```python
raw_datasets["train"].column_names
```




    ['tokens', 'ner_tags', 'langs', 'spans']



We have to now pad our data input sequences and their labels to the proper input size.


```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    return_tensors='tf'
)
```


```python
BATCH_SIZE = 16

tf_train_dataset = tokenized_datasets['train'].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=BATCH_SIZE
)

tf_validation_dataset = tokenized_datasets['validation'].to_tf_dataset(
    columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=BATCH_SIZE
)


```

## 4. The BERT model



We'll be using the `TFAutoModelForTokenClassification` class, which allows us to load in different types of pre-trained models like BERT and GPT-3 while providing a unified interface. We also don't have to define the input shape, we just have to pass a mapping between ID and their label.


```python
id2label = { i: label for i, label in enumerate(label_names) }
id2label
```




    {0: 'O',
     1: 'B-PER',
     2: 'I-PER',
     3: 'B-ORG',
     4: 'I-ORG',
     5: 'B-LOC',
     6: 'I-LOC'}




```python
label2id = { v: k for k, v in id2label.items() }
label2id
```




    {'O': 0,
     'B-PER': 1,
     'I-PER': 2,
     'B-ORG': 3,
     'I-ORG': 4,
     'B-LOC': 5,
     'I-LOC': 6}




```python
from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id
)
```


## 5. Finetuning the model with our labels


Let's first set some of the hyperparameters necessary for the finetuning, and let's use a optimizer.


```python
from transformers import create_optimizer
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16') # chop weights in half for speed gain

NUM_EPOCHS = 3
num_train_steps = len(tf_train_dataset) * NUM_EPOCHS

optimizer, schedule = create_optimizer(
    init_lr=2e-5, # learning rate at the end of warmup
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01 # decay of steps as steps increase
)

model.compile(optimizer=optimizer)
```


```python
model.summary()
```

    Model: "tf_bert_for_token_classification"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     bert (TFBertMainLayer)      multiple                  108546816 
                                                                     
     dropout_37 (Dropout)        multiple                  0 (unused)
                                                                     
     classifier (Dense)          multiple                  5383      
                                                                     
    =================================================================
    Total params: 108552199 (414.09 MB)
    Trainable params: 108552199 (414.09 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________


Let's train the model and push it to HuggingFace during training.


```python
from transformers.keras_callbacks import PushToHubCallback

push_to_hf = PushToHubCallback(
    output_dir='bert-finetuned-ner-nl-wiki',
    tokenizer=tokenizer
)

model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    callbacks=[push_to_hf],
    epochs=NUM_EPOCHS,
)
```

## 7. Evaluate model
Now that our model has been trained over three epochs, we can check out its performance against the validation dataset using `seqeval`.


```python
!pip install seqeval evaluate
```

```python
import evaluate

metric = evaluate.load('seqeval')
```


```python
import numpy as np

all_predictions = []
all_labels = []
for batch in tf_validation_dataset:
    logits = model.predict_on_batch(batch)["logits"]
    labels = batch["labels"]
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(label_names[predicted_idx])
            all_labels.append(label_names[label_idx])
metric.compute(predictions=[all_predictions], references=[all_labels])
```




    {'LOC': {'precision': 0.8893850213111427,
      'recall': 0.9063081695966908,
      'f1': 0.8977668510551117,
      'number': 4835},
     'ORG': {'precision': 0.8211243611584327,
      'recall': 0.8556936342886128,
      'f1': 0.838052657724789,
      'number': 3943},
     'PER': {'precision': 0.9298908480268682,
      'recall': 0.9318468657972234,
      'f1': 0.9308678293759193,
      'number': 4754},
     'overall_precision': 0.8830434782608696,
     'overall_recall': 0.9005320721253326,
     'overall_f1': 0.8917020342455729,
     'overall_accuracy': 0.9652627575880018}



We can see that the weighted overall precision, recall, and f1 are rather high, indicating that our model performs rather well on its limited amount of training. We can definitely improve the model by trying to find more data, use data augmentation or train the model longer, though we should be careful with overfitting.

## 8. Testing our model
Since we've pushed the model to HuggingFace, we can now pull it and use it.


```python
from transformers import pipeline

model_checkpoint = 'pavlad/bert-finetuned-ner-nl-wiki'
token_classifier = pipeline(
    'token-classification',
    model=model_checkpoint,
    aggregation_strategy='simple' # this will group the matched entities
)

```


```python
token_classifier("In reactie op het tekort aan muntstukken biedt de bankenfederatie Febelfin een oplossing. “Je kunt de munten ook naar je bankkantoor brengen”, zegt Isabelle Marchand van Febelfin.")
```




    [{'entity_group': 'ORG',
      'score': 0.799,
      'word': 'Febelfin',
      'start': 66,
      'end': 74},
     {'entity_group': 'PER',
      'score': 0.988,
      'word': 'Isabelle Marchand',
      'start': 148,
      'end': 165},
     {'entity_group': 'ORG',
      'score': 0.6245,
      'word': 'Febelfin',
      'start': 170,
      'end': 178}]



# 9. Conclusion

We were able to create a rather well-performing BERT model to perform Named Entity Recognition on Dutch documents. This demonstrates the power of the BERT architecture, especially when leveraging models that have been pretrained by larger organizations.
