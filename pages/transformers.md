# Transformer architecture from scratch

# 1. Intro
## 1.1 Why transformers?
Before the *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)* paper was released in 2017, LSTMs were the de facto standard for seq2seq modeling. These models were struggling on two main points:

1. **Vanishing gradients**: LSTMs process input sequentially, and especially for longer sequences, they would lose track of the first inputs.
2. **Slowness in training**: Since LSTM networks are fed inputs sequentially, always having to rely on the previous output during training, it would take a lot of time to train a network as there was no way of parallelising the process.

Transformers solved many of the problems LSTMs had using a new attention mechanism:
1. **No more vanishing gradients**: This mechanism allowed the network to learn which inputs related to other inputs, allowing for a theoretically unlimited input as the network doesn't have to focus on ALL the inputs.
2. **Concurrency**: As the attention mechanism doesn't rely on sequences anymore, it was possible to parallelise and thus speed up the training process.

The new model also included bidirectional encoding, which allowed the model to understand the context of words better, as it would look left-to-right at the inputs, and vice-versa.

## 1.2 Objective
During my studies, we have looked at many deep learning architectures, including CNNs, RNNs (LSTMs), and transformers, but most of it was rather theoretical. This is why I wanted to write the transformer architecture more or less from scratch to solidify my knowledge and gain a better intuition of this type of network.

I have obviously not figured all of this out by myself, but have used many resources to gain a better understanding. I have mainly followed the [TensorFlow transformer tutorial](https://www.tensorflow.org/text/tutorials/transformer), but I often took a break to consult other resources, for which I've added notes here.

Here's a non-exhaustive list of videos and articles I've used:

1. https://peterbloem.nl/blog/transformerslist=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
2. https://jalammar.github.io/illustrated-transformer/
3. https://www.tensorflow.org/text/tutorials/transformer
4. https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/
5. https://www.youtube.com/watch?v=bCz4OMemCcA
6. https://www.youtube.com/watch?v=aircAruvnKk&

# 2. Subcomponents of the transformer
<div align="center">
<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="drawing" width="400"/><br>
Source: tensorflow.org
</div>

The transformer architecture consists of two main parts: the **encoder** and the **decoder**.

In the context of NLP and seq2seq modeling, the encoder is responsible for understanding the input language, while the decoder incorporates the target language and combines the information from the encoder to predict the next token.

The encoder and decoder consist of smaller parts, which we'll develop one by one to finally be able to construct a full transformer model.

Let's first import the necessary dependencies:
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
```


## 2.1 Positional embedding

In this step, input tokens are converted to vectors with many dimensions, representing the tokens in high-dimensional space. This allows for vector arithmetic, and a typical example of this is `king - man + woman = queen`. To do this, we'll use the Keras embedding layer `tf.keras.layers.Embedding`.

If we input the model the tokens without an indication of their order, then the transformer would essentially follow the bag-of-words approach, where it wouldn't learn much about how phrases are structured. To overcome this problem, we make use of positional encoding, which is essentially a vector of alternating sine and cosine functions with different frequencies that are added to the embeddings. This allows the model to capture the long and short-term relationships between words.

In the *Attention Is All You Need* paper, the following functions are used for the positional encoding:
- $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$
- $PE_{(pos, 2i + 1)} = \cos(pos/10000^{2i/d_{model}})$

Where:
- $pos$ is the position or index within the input sequence
- $d_{model}$ is the number of output dimensions of the model
- $10000$ is the scaling factor that adjusts the frequency, taken from the *Attention Is All You Need* paper
- $2i$ and $2i + 1$ simply encode $\sin$ for even and $\cos$ for odd inputs

For simplicity's sake, we're concatenating the sines and cosines instead of interleaving them like in the original paper, but this solution is equivalent to the original.


```python
def positional_encoding(length, depth):
    depth = depth/2 # dividing by two as we both use cosine and sine

    # using np.newaxis so to that we can apply broadcasting to otherise incompatible matrices and vectors
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
    [np.sin(angle_rads), np.cos(angle_rads)],
    axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)
```

Now that we have the embedding layer and custom code to positionally encode sequences, we can combine them in a new positional embedding layer.

```python
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,  # number of tokens
            d_model,  # number dimensions for embeddings
            mask_zero=True  # ignores zeroes used for padding
        )
        self.pos_encoding = positional_encoding(length=vocab_size, depth=d_model)

    # makes sure that padding tokens are ignored, which improves the accuracy of the model and performance
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        # calculate embedding and corresponding positional encoding
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # normalizing the scale of the embedding
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x
```

## 2.2 Attention

The Multi-Headed Attention block is one of the most important parts of the transformer architecture, as it allows the model to train and learn which token is related to which other token in the input sequence. During the training process, the attention scores are updated so that the model learns to focus on the most relevant words in the input sequence.

At first, three sets of vectors with randomized weights $W_Q$, $W_K$, and $W_V$ are created. "Multi-headed" refers to the number of these sets of vectors used simultaneously. Similar to CNN filters, these different heads focusing on the same inputs allow the model to pick up more details. The vectors are stacked to form matrices and are multiplied with the tokenized input sequence $X$ to produce three new matrices with the shape **(number of tokens, number of embeddings)**:
- $Query = XW_Q$
- $Key = XW_K$
- $Value = XW_V$

These three matrices are used to calculate the attention scores. Simply put, this layer uses three vectors, which are initially randomized:
1. **query**: contains what we're trying to find
2. **key**: contains what kind of information we have
3. **value**: contains the information we have

These three vectors are used to calculate the attention scores with:

$Attention(Query, Key, Value) = softmax(\dfrac{QueryKey^T}{\sqrt{dimension_{Key}}}) Value$

- The dot product $QueryKey^T$ computes a score for each token in the key sequence with respect to the query.
- $\sqrt{dimension_{Key}}$ scales the dot product to stabilize the gradients during training.
- $softmax$ converts the scores into probabilities that sum to 1.

Once the attention scores are calculated for each head, they are concatenated, a linear transformation is performed, and the result is fed to the next layer.

We'll first build the attention layer based on the formula above.

```python
from tensorflow.keras.layers import Layer
from tensorflow import matmul, math, cast, float32
from keras.backend import softmax

class DotProductAttention(Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, queries, keys, value, d_k, mask=None):
    scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

    # ignoring attention score for masked token
    if mask is not None:
      scores += -1e9 * mask

    weights = softmax(scores)

    return matmul(weights, values)
```

Now we build the Multi-head Attention Layer. The method `reshape_tensor` is crucial because during the forward pass it reshapes and transposes the input to prepare for multi-head attention, and during the backward pass it reshapes the tensor back to its original shape.

```python
class MultiHeadAttention(Layer):
  def __init__(self, h, d_k, d_v, d_model, **kwargs):
    super().__init__(**kwargs)
    self.attention = DotProductAttention()
    self.heads = h # number of heads
    self.d_k = d_k # dimensions of Key and Query vectors
    self.d_v = d_v # dimensions of Value vectors
    self.d_model = d_model # dimension of input values e.g. (max input sequence, number of dimensions)
    self.W_q = Dense(d_k)
    self.W_k = Dense(d_k)
    self.W_v = Dense(d_v)
    self.W_o = Dense(d_model) # projection matrix for output

  # reshapes tensor for forward and backward pass
  def reshape_tensor(self, x, heads, flag):
    if flag:
      x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
      x = transpose(x, perm=(0, 2, 1, 3))
    else:
      x = transpose(x, perm=(0, 2, 1, 3)) # reverting transpose
      x = reshape(x, shape=(shape(x)[0], shape(x)[1], -1))

    return x

  def call(self, queries, keys, value, mask=None):
    # rearrange Q, K and V to compute them in parallel
    q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
    k_reshaped = self.reshape_tensor(self.W_k(queries), self.heads, True)
    v_reshaped = self.reshape_tensor(self.W_v(queries), self.heads, True)

    # calculate output based on the reshaped tensors
    o_reshaped = self.reshape_tensor(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)

    output = self.reshape_tensor(o_reshaped, self.heads, False) # concatenate output

    # apply linear transform
    return self.W_o(output)



```

Now that we have the multi-head attention layer, we can use it as a base for the three types of attention layers present in the transformer architecture.


```python
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization() # adds stability to the network by normalizing the output
    self.add = tf.keras.layers.Add()
```

### 2.2.1 Cross-attention



<div align="center">
<img src="https://www.tensorflow.org/images/tutorials/transformer/CrossAttention-new-full.png" alt="drawing" width="400"/><br>
Source: tensorflow.org
</div>

This attention mechanism connects the **key** and **value** vectors from the encoder (e.g., source language) with the **query** vector from the decoder (e.g., target language). The output of the layer has the length of the **query** vector.

```python
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
      query=x,
      key=context,
      value=context,
      return_attention_scores=True
    )

    self.last_attn_scores = attn_scores # used for evaluation

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
```

### 2.2.2 Self-attention

This layer is the self-attention layer used in the encoder, where it tries to understand the context of the input sequence and how each token is related to the others.

```python
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
      query=x,
      value=x,
      key=x
    )
    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
```

### 2.2.3 Masked attention

Since our model tries to predict the next token, we have to make sure that the decoder only depends on the previous sequence elements. This prevents data leakage and allows the model to train more efficiently by processing the input sequence up to the token where the mask is applied.

```python
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
      query=x,
      value=x,
      key=x,
      use_causal_mask = True
    )
    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
```

## 2.3 Feed-forward network

Both the encoder and decoder include a feed-forward network. It introduces non-linearity into the network by applying the ReLU activation function. This allows the network to grasp more complexity, which is fundamental to deep learning. It also contains a dropout layer which helps prevent the model from overfitting.

```python
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)

    return x
```

# 3. The encoder
We can now combine the input embedding and positional encoding layer with the encoder layer.

```python
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
      num_heads=num_heads,
      key_dim=d_model,
      dropout=dropout_rate
    )

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
```

```python
class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
      vocab_size=vocab_size,
      d_model=d_model
    )

    self.enc_layers = [
      EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate
      ) for _ in range(num_layers)
    ]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    x = self.pos_embedding(x)
    x = self.dropout(x) # used to prevent overfitting

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x
```

# 4. The decoder

We can now also construct our decoder layer, which consists of the masked attention layer, cross-attention layer, and feed-forward network.

```python
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)

    return x
```


```python
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    x = self.pos_embedding(x)
    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    return x
```

# 5. The transformer

We put together the encoder and decoder components, and attach the linear output layer, which is used to generate the final predictions, using softmax activation to produce the probability distribution over the target token.

```python
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(
      num_layers=num_layers, d_model=d_model,
      num_heads=num_heads,
      dff=dff,
      vocab_size=input_vocab_size,
      dropout_rate=dropout_rate
    )

    self.decoder = Decoder(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dff=dff,
      vocab_size=target_vocab_size,
      dropout_rate=dropout_rate
    )

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    context, x  = inputs # matches Keras .fit function
    context = self.encoder(context)
    x = self.decoder(x, context)
    logits = self.final_layer(x) # final output

    try:
      del logits._keras_mask # prevent scaling losses
    except AttributeError:
      pass
    return logits
```

# 6. Testing our model

We'll test our model on a traditional seq2seq task: translation. In this case, we'll look at translating Portuguese to English.

## 6.1 Dataset

We'll use a dataset that is available through TensorFlow.

```python
examples, metadata = tfds.load(
  'ted_hrlr_translate/pt_to_en',
  with_info=True,
  as_supervised=True
)

train_examples, val_examples = examples['train'], examples['validation']
```

## 6.2 Tokenizer
We need a tokenizer to give each word or subword a numeric representation, which:
- splits punctuation
- lowercases text
- normalizes text to UTF-8
- splits unknown words into subwords that it does know

We'll use one that has been created in the tutorial: https://www.tensorflow.org/text/guide/subwords_tokenizer.


```python
model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/ted_hrlr_translate_pt_en_converter.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)
```

## 6.3 Data pipeline

Our data pipeline will convert the input text into ragged batches, which are batches of sequences where each one can have a different length. Inputs are trimmed to the maximum number of tokens.

We also manipulate the inputs and the labels:
- `inputs`: input sequence fed to the model, includes the start token only
- `labels`: target sequence that the model should predict, includes the end token only

```python
MAX_TOKENS = 128

def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    pt = pt[:, :MAX_TOKENS]
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()
    en_labels = en[:, 1:].to_tensor()

    return (pt, en_inputs), en_labels
```

We convert the datasets into data of batches for training.

```python
BUFFER_SIZE = 20_000  # the larger the buffer, the more randomized the order is
BATCH_SIZE = 64
```
```python
def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)  # makes batch available when needed
    )
```
```python
total_train_size = tf.data.experimental.cardinality(train_examples).numpy()
train_examples_half = train_examples.take(total_train_size//3)
train_batches = make_batches(train_examples_half)
val_batches = make_batches(val_examples)
```

```python
for (pt, en), en_labels in train_batches.take(1):
  break
```

## 6.4 Optimizer

We use the Adam optimizer with the same parameters as in the original *Attention is All You Need* paper.

$$
learningRate = modelDimension^{-0.5} \cdot \min(stepNum^{-0.5}, stepNum \cdot warmupSteps^{-1.5})
$$

Where:

- $\beta_1 = 0.9$
- $\beta_2 = 0.98$
- $\epsilon = 10^{-9}$
- $warmupSteps = 4000$

The idea behind the optimizer is that it will try to find the global minimum, and it's designed to adjust the learning rate based on the step number and its parameters.

- $modelDimension^{-0.5}$ scales the learning rate based on the dimensionality of the input embeddings
- $stepNum^{-0.5}$ decreases the learning rate over time, trying to prevent overshooting the minimum
- $stepNum \cdot warmupSteps^{-1.5}$ initially starts the model with a small learning rate and then increases it over time

```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4_000):
    super().__init__()

    self.d_model = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```

## 6.5 Metrics

Since we are masking the labels we have to predict in the decoder, we need to ensure that we ignore any loss coming from any masked label, as we'd have a data leak otherwise.

```python
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss
```

We have to do the same thing for the accuracy:

```python
def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0  # we ignore the padding token that has position of 0
    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)
```

## 6.6 Training the model

We have to define the model before training it, specifying all of the hyperparameters. I've tried to minimize the complexity of the model as I'm working with rather limited hardware.

```python
LAYERS = 3 # depth of model, was 4 in paper
INPUT_TOKEN_DIMENSIONS = 128
FEED_FORWARD_DIMENSIONS = 512
HEADS = 4 # was 8 in paper
DROPOUT_RATE = 0.1
```

```python
transformer = Transformer(
  num_layers=LAYERS,
  d_model=INPUT_TOKEN_DIMENSIONS,
  num_heads=HEADS,
  dff=FEED_FORWARD_DIMENSIONS,
  dropout_rate=DROPOUT_RATE,
  input_vocab_size=tokenizers.pt.get_vocab_size().numpy(), # input layer
  target_vocab_size=tokenizers.en.get_vocab_size().numpy() # output layer
)

transformer.summary()
```

We compile the model.

```python
transformer.compile(
  loss=masked_loss,
  optimizer=optimizer,
  metrics=[masked_accuracy]
)
```

Now we train it - this took a LONG while.

```python
transformer.fit(
    train_batches,
    epochs=2, # again very limited due to hardware constraints
    validation_data=val_batches
)
```

    Epoch 1/2
    270/270 ━━━━━━━━━━━━━━━━━━━━ 5792s 21s/step - loss: 8.5714 - masked_accuracy: 0.0374 - val_loss: 7.1137 - val_masked_accuracy: 0.1253
    Epoch 2/2
    270/270 ━━━━━━━━━━━━━━━━━━━━ 2719s 10s/step - loss: 6.6087 - masked_accuracy: 0.1508 - val_loss: 5.5630 - val_masked_accuracy: 0.2077
    <keras.src.callbacks.history.History at 0x79781c401600>

And we can make sure to save the Keras model.

```python
transformer.save('pt_to_en_demo.keras')
```

## 6.7 Testing the model

Here I'll take the code straight from the tutorial, as it simply serves for testing the model.

```python
class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # As the output language is English, initialize the output with the
    # English `[START]` token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

    tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    return text, tokens, attention_weights
```

We can finally apply the translation and print it.

```python
translator = Translator(tokenizers, transformer)
```

```python
def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')
```

```python
sentence = 'Gosto de ananás na pizza.'
ground_truth = 'I like pineapple on pizza.'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)
```

    Input:         : Gosto de ananás na pizza.
    Prediction     : and the .
    Ground truth   : I like pineapple on pizza.

Well, that was rather disappointing but not surprising: our network works, but due to having to minimize the hyperparameters, it obviously didn't perform well.

# 7. Conclusion

We have built part of a transformer neural net from scratch, and even though the performance wasn't amazing due to the lack of training resources, I think I gained a lot of insight from writing the code, digging deeper into certain concepts, and connecting all of the components.

In terms of evaluating the results, I'd like to apply the [BLEU](https://keras.io/api/keras_nlp/metrics/bleu/) metric once I get a better working model. This metric evaluates the matching number of N-grams between the predicted and model translations.

I think that I would want to repeat this exercise later on, once I've gained more experience with neural networks, to really solidify my knowledge. It would also be interesting to build a tokenizer and the embedding layer from scratch.
