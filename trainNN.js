const tf = require('@tensorflow/tfjs');
const data = require('./data.json');

// Define the model
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [data.inputs[0].length], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 16, activation: 'relu'}));
model.add(tf.layers.dense({units: data.outputs[0].length, activation: 'softmax'}));

// Compile the model
model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});

// Shuffle the data
const inputs = data.inputs.slice();
const outputs = data.outputs.slice();
for (let i = inputs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [inputs[i], inputs[j]] = [inputs[j], inputs[i]];
    [outputs[i], outputs[j]] = [outputs[j], outputs[i]];
}

// Split the data into training and testing sets
const trainSize = Math.floor(inputs.length * 0.8);
const trainInputs = inputs.slice(0, trainSize);
const trainOutputs = outputs.slice(0, trainSize);
const testInputs = inputs.slice(trainSize);
const testOutputs = outputs.slice(trainSize);

// Train the model
const history = await model.fit(tf.tensor2d(trainInputs), tf.tensor2d(trainOutputs), {
    epochs: 100,
    validationData: [tf.tensor2d(testInputs), tf.tensor2d(testOutputs)],
    callbacks: {
        onEpochEnd: async (epoch, logs) => {
            console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
        }
    }
});

// Evaluate the model on the test data
const testResult = model.evaluate(tf.tensor2d(testInputs), tf.tensor2d(testOutputs));
console.log(`Test loss: ${testResult[0].dataSync()}, Test accuracy: ${testResult[1].dataSync()}`);
