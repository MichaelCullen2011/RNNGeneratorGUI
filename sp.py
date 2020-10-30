import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time


'''
Logic
'''
run = False
Train = False
Load = True
Generate = True
training_step = False
loading_step = False
generate_step = False
EPOCHS = 50

'''
Functions
'''


def start():
    global training_step, loading_step, generate_step
    if run:
        print("Running SP NN")
        '''
        TXT File Input, Editing and Parsing
        '''
        path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

        vocab = sorted(set(text))
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)

        text_as_int = np.array([char2idx[c] for c in text])

        seq_length = 100
        examples_per_epoch = len(text)//(seq_length+1)

        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        dataset = sequences.map(split_input_target)
        BATCH_SIZE = 64
        BUFFER_SIZE = 10000
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        vocab_size = len(vocab)
        embedding_dim = 256
        rnn_units = 1024

        def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
                tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    stateful=True,
                                    recurrent_initializer='glorot_uniform'),
                tf.keras.layers.Dense(vocab_size)
                                    ])
            return model

        model = build_model(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            batch_size=BATCH_SIZE)

        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)

        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer='adam', loss=loss)

        '''
        Training
        '''
        checkpoint_dir = 'training_checkpoints/SP'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        @tf.function
        def train_step(inp, target):
            with tf.GradientTape() as tape:
                predictions = model(inp)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return loss

        if Train:
            def training_step():
                global model
                training_step = True
                # Training step
                for epoch in range(EPOCHS):
                    start = time.time()

                    # resetting the hidden state at the start of every epoch
                    model.reset_states()

                    for (batch_n, (inp, target)) in enumerate(dataset):
                        loss = train_step(inp, target)

                        if batch_n % 100 == 0:
                            template = 'Epoch {} Batch {} Loss {}'
                            print(template.format(epoch + 1, batch_n, loss))

                    # saving (checkpoint) the model every 5 epochs
                    if (epoch + 1) % 5 == 0:
                        model.save_weights(checkpoint_prefix.format(epoch=epoch))

                    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
                    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

                model.save_weights(checkpoint_prefix.format(epoch=epoch))
                print(checkpoint_dir)
                tf.train.latest_checkpoint(checkpoint_dir)
                model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
                model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
                model.build(tf.TensorShape([1, None]))
                model.save('./models/SP')
                training_step = False

        '''
        Loading Model
        '''
        if Load:
            loading_step = True
            model = keras.models.load_model('models/SP')
            loading_step = False

        if Generate:
            generate_step = True

            def generate_text(model, start_string):
                num_generate = 1000
                input_eval = [char2idx[s] for s in start_string]
                input_eval = tf.expand_dims(input_eval, 0)
                text_generated = []
                temperature = 1.0

                model.reset_states()
                for i in range(num_generate):
                    predictions = model(input_eval)
                    predictions = tf.squeeze(predictions, 0)
                    predictions = predictions / temperature
                    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

                    input_eval = tf.expand_dims([predicted_id], 0)

                    text_generated.append(idx2char[predicted_id])
                return start_string + ''.join(text_generated)
            print(generate_text(model, start_string=u"ROMEO: "))
            generate_step = False
