const Matrix = require("./matrix").Matrix;
const GPU = require("gpu.js").GPU;
const gpu = new GPU({
    mode: 'cpu'
});
const M = Matrix.getFunctions(gpu);

/* function t() {
    return Date.now();
}

function tdiff(t1) {
    console.log(Date.now() - t1);
    return Date.now() - t1;
} */


class NeuralNetwork {
    constructor(inputs_n, hidden_n, outputs_n, learningRate = 0.1) {
        this.layers = []
        // Initialise input layer as a vector
        this.layers.push(new Matrix(null, inputs_n, 1).withBias());

        // Now initialise hidden layer(s) as a vector or array of vectors
        // hidden_n can either be a number of hidden nodes or an array of hidden nodes per layer so..
        if (typeof hidden_n == "number") {
            this.layers.push(new Matrix(null, hidden_n, 1).withBias());
        } else {
            for (let hlayer = 0; hlayer < hidden_n.length; hlayer++) {
                this.layers.push(new Matrix(null, hidden_n[hlayer], 1).withBias());
            }
        }

        // Initialise output layer as a vector  - NOTE no bias is needed for the output layer
        this.layers.push(new Matrix(null, outputs_n, 1));

        // Initialise weights array
        this.weights = [];
        // Then fill with dummy weight matrices
        for (let i = 0; i < this.layers.length - 2; i++) {
            // Create new matrix where rows = no of nodes on 'output side' and cols = no of nodes on 'input side'
            this.weights.push(new Matrix(null, this.layers[i + 1].rows, this.layers[i].rows).randomFill()); // -1 rows because of bias on input/hidden layers
        }

        this.weights.push(new Matrix(null, this.layers[this.layers.length - 1].rows, this.layers[this.layers.length - 2].rows).randomFill());


        // Set learning rate property
        this.learningRate = learningRate;
    }

    feedForward(inputs_arr) {
        // set INPUT layer from array
        this.layers[0] = M.fromArray(inputs_arr).withBias();

        //set remaining layer(s) =previous layer * `input` weights
        for (let layer = 1; layer < this.layers.length - 1; layer++) {
            /* console.log("FF layer ", layer);
            this.weights[layer - 1].print();
            this.layers[layer - 1].print(); */
            this.layers[layer] = M.mapSigmoid(M.dotProduct(this.weights[layer - 1], this.layers[layer - 1])).setBias();
        }
        this.layers[this.layers.length - 1] = M.mapSigmoid(M.dotProduct(this.weights[this.layers.length - 2], this.layers[this.layers.length - 2]));

        return this.layers[this.layers.length - 1];
    }

    backProp(inputs_arr, targets_arr) {
        //console.log(inputs_arr, targets_arr);
        // First feed forward to gauge errors
        let outputs = this.feedForward(inputs_arr);

        let targets = M.fromArray(targets_arr);
        let errors = M.subtract(outputs, targets);

        let gradients = M.hadProduct(M.mapSigmoid_P(outputs), errors);
        gradients = M.scalarProduct(gradients, this.learningRate);
        let deltaWeights = M.dotProduct(
            gradients,
            M.transpose(this.layers[this.layers.length - 2]),
        );
        this.weights[this.weights.length - 1] = M.sum(this.weights[this.weights.length - 1], deltaWeights);
        //console.log("successful backprop of first layer");

        for (let layer = this.layers.length - 2; layer > 0; layer--) {
            //console.log("trying to backprop layer: ", layer);
            errors = M.dotProduct(M.transpose(this.weights[layer]), errors);

            gradients = M.hadProduct(M.mapSigmoid_P(this.layers[layer]), errors);
            gradients = M.scalarProduct(gradients, this.learningRate);
            //gradients.print();

            deltaWeights = M.dotProduct(
                gradients,
                M.transpose(this.layers[layer - 1]),
            );

            //this.print();
            //M.transpose(this.layers[layer - 1]).print();

            //this.weights[layer - 1].print()
            //deltaWeights.print();
            this.weights[layer - 1] = M.sum(this.weights[layer - 1], deltaWeights);

            //console.log("completed layer ", layer)

        }

    }

    print() {
        console.log("=====================================================================")
        this.layers.forEach((layer, i) => {
            console.log("Layer: " + i);
            console.table(layer.d);
        })
        this.weights.forEach((layer, i) => {
            console.log("Layer " + i + "->" + (i + 1) + " weights:");
            console.table(layer.d);
        })
        console.log("Learning rate = ", this.learningRate);
        console.log("=====================================================================")
    }
}

module.exports = {
    NeuralNetwork,
};