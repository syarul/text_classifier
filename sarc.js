import tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import { pad_sequences } from 'array-sequence-utils'
import Tokenizer from 'text-tokenizer-utils'

import { readFileSync } from 'fs'

// Define constants for configuration
const vocab_size = 10000
const embedding_dim = 16
const max_length = 100
const trunc_type = 'post'
const padding_type = 'post'
const oov_tok = '<OOV>'
const training_size = 20000

// Read and parse the dataset
const datastore = JSON.parse(readFileSync('./tmp/sarcasm.json'))

// Separate sentences and labels from the dataset
const sentences = []
const labels = []

for (const item of datastore) {
  sentences.push(item.headline)
  labels.push(item.is_sarcastic)
}

// Split the dataset into training and testing sets
const training_sentences = sentences.slice(0, training_size)
const testing_sentences = sentences.slice(training_size)
let training_labels = labels.slice(0, training_size)
let testing_labels = labels.slice(training_size)

// Initialize a tokenizer with the specified vocabulary size and out-of-vocabulary token
const tokenizer = new Tokenizer(vocab_size, oov_tok)

// Fit the tokenizer on the training sentences to build the word index
tokenizer.fit_on_texts(training_sentences)

// Check the vocab word_index
// console.log(tokenizer.word_index())

// Convert the training and testing sentences to sequences of tokens
const training_sequences = tokenizer.texts_to_sequences(training_sentences)
let training_padded = pad_sequences(training_sequences, max_length, padding_type, trunc_type)
const testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
let testing_padded = pad_sequences(testing_sequences, max_length, padding_type, trunc_type)

// Convert the padded sequences and labels to TensorFlow tensors
training_padded = tf.tensor2d(training_padded)
training_labels = tf.tensor2d(training_labels, [training_labels.length, 1])
testing_padded = tf.tensor2d(testing_padded)
testing_labels = tf.tensor2d(testing_labels, [testing_labels.length, 1])

// Define the model architecture
const model = tf.sequential()

model.add(tf.layers.embedding({ inputDim: vocab_size, outputDim: embedding_dim, inputLength: max_length }))
model.add(tf.layers.globalAveragePooling1d())
model.add(tf.layers.dense({ units: 24, activation: 'relu' }))
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }))
model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['accuracy'] })

// Print a summary of the model architecture
model.summary()

// Train the model on the training data
const num_epochs = 30

await model.fit(training_padded, training_labels, { 
  epochs: num_epochs,
  validationData: [testing_padded, testing_labels],
  verbose: 2
})

// Use the trained model to make predictions on new sentences
const new_sentences = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
const sequences = await tokenizer.texts_to_sequences(new_sentences)
const padded = pad_sequences(sequences, max_length, padding_type, trunc_type)

for (const p of padded) {
  const sentenceTensor = tf.tensor2d(p, [1, max_length])
  const predictions = model.predict(sentenceTensor)
  const weight = predictions.dataSync()[0]
  console.log('\n', weight, weight.toExponential())
}
