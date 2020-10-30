import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import time
from shutil import copyfile


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
        print("Train", Train, "Generate", Generate)
        print("Running HP NN")
        '''
        TXT File Input, Editing and Parsing
        '''
        file_name = 'Harry Potter Complete.txt'
        os_path = 'C:/Users/micha/.keras/datasets/'
        abs_path = str(os_path + file_name)
        edited_path = 'edited ' + file_name

        def edit_txt_file(path):    # Edits the raw txt into a single line (line breaks added in generation)
            with open(path, 'r', encoding="utf8") as HP:
                with open('copy ' + file_name, 'r+', encoding="utf8") as HP_copy:
                    copyfile(path, 'copy ' + file_name)
                    with open('edited ' + file_name, 'w', encoding="utf8") as HP_edited:
                        string_without_line_breaks = ""
                        for line in HP_copy:
                            stripped_line = line.replace('\n', ' ')
                            string_without_line_breaks += stripped_line
                        HP_edited.write(string_without_line_breaks)
                        # count = 0
                        # top_end = 150
                        # try:
                        #     while count < len(string_without_line_breaks):
                        #         for j in range(0, top_end):
                        #             HP_edited.write(string_without_line_breaks[j + count])
                        #         if string_without_line_breaks[count + top_end] != ' ':
                        #             HP_edited.write('-' + '\n' + '-')
                        #         else:
                        #             HP_edited.write('\n')
                        #         count += 150
                        # except IndexError:
                        #     pass

        path_to_file = tf.keras.utils.get_file(file_name, 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        edit_txt_file(abs_path)

        text = open(edited_path, 'rb').read().decode(encoding='utf-8')

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

        '''
        Build Empty Model
        '''

        def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size,
                                          embedding_dim,
                                          batch_input_shape=[batch_size, None]),
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
        checkpoint_dir = 'training_checkpoints/HP/'
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

        def training_model():
            if Train:
                model = build_model(
                    vocab_size=len(vocab),
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units,
                    batch_size=BATCH_SIZE)
                global training_step
                training_step = True
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
                model.save('./models/HP')
                training_step = False

        '''
        Loading Model
        '''
        def loading_model():
            global loading_step
            model = build_model(
                vocab_size=len(vocab),
                embedding_dim=embedding_dim,
                rnn_units=rnn_units,
                batch_size=BATCH_SIZE)
            if Load:
                loading_step = True
                model = keras.models.load_model('models/HP')
                loading_step = False

        '''
        Generate Text
        '''
        def generate_step():
            global generate_step
            if Generate:
                generate_step = True

                def generate_text(model, start_string):
                    num_generate = 1000
                    input_eval = [char2idx[s] for s in start_string]
                    input_eval = tf.expand_dims(input_eval, 0)
                    text_generated = []
                    temperature = 0.9

                    model.reset_states()
                    count = 150
                    for i in range(num_generate):
                        if i == count:
                            if idx2char[predicted_id] == ' ':
                                text_generated.append('\n')   # appends a line break every 150 characters on a space
                            elif idx2char[predicted_id] != ' ':
                                text_generated.append('- \n -')  # appends a dash break dash every 150 characters on a split word
                            count += 150
                        predictions = model(input_eval)
                        predictions = tf.squeeze(predictions, 0)
                        predictions = predictions / temperature
                        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

                        input_eval = tf.expand_dims([predicted_id], 0)

                        text_generated.append(idx2char[predicted_id])
                    return start_string + ''.join(text_generated)

                start_string = input("Type a starting string: ")
                print(generate_text(model, start_string=start_string))
                generate_step = False

