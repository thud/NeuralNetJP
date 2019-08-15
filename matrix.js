class Matrix {
    constructor(data_, rows_, cols_, biased_) {
        // initialise Matrix objects (data = 2d array, rows = no. of rows, cols = no. of cols)
        this.d = data_;
        this.rows = rows_;
        this.cols = cols_;
        this.biased = biased_;

        if ((!this.rows || !this.cols) && !this.d)
            throw new Error("Matrix constructor requires either data as a 2d array or rows/cols lengths");

        if (!this.d) { // if no given starting array then...
            // setup a new array full of zeroes
            this.d = new Array(this.rows);
            for (let row = 0; row < this.d.length; row++) {
                this.d[row] = new Array(this.cols);
            }
            this.fill(0);
        } else {
            this.rows = this.d.length;
            this.cols = this.d[0].length;
        }

        if (this.biased) {
            this.withBias();
        }
    }

    fill(n) {
        for (let x = 0; x < this.d.length; x++) {
            for (let y = 0; y < this.d[x].length; y++) {
                this.d[x][y] = n;
            }
        }
        return this;
    }

    randomFill(min = 0.45, max = 0.55) {
        for (let x = 0; x < this.d.length; x++) {
            for (let y = 0; y < this.d[x].length; y++) {
                this.d[x][y] = (Math.random() * (max - min)) + min;
            }
        }
        return this;
    }

    withBias() {
        // adds a new first row with a value of 1
        this.biased = true;
        return new Matrix(
            this.d.concat([
                [1]
            ]));
    }

    setBias() {
        this.d[this.d.length - 1] = this.d[0].map(_n => 1);
        return this;
    }

    /*  withoutBias() {
         // adds a new first row with a value of 1
         this.biased = false;
         return new Matrix(
             this.d.pop()
     } */

    print() {
        console.table(this.d);
    }

    map() {} // don't use for application of activation functions as theoretically much slower

    static getFunctions(gpu) {
        // Create kernel functions and return to the requestor
        const dotProductKernel = (settings) => gpu.createKernel(function (a, b) {
            let sum = 0;
            for (let i = 0; i < this.constants.ySize; i++) {
                sum += a[this.thread.y][i] * b[i][this.thread.x];
            }
            return sum;
            /* let sum = 0;
            for (let i = 0; i < a.length; i++) {
                sum += a[this.thread.y][i] * b[i][this.thread.x];
            }
            return sum; */
        }, settings);

        const dotProduct = (a, b) => {
            if (a.cols !== b.rows) {
                a.print();
                b.print();
                throw "DOTPRODUCT: cols of mat a does not equal rows of b";
            }
            return new Matrix(dotProductKernel({
                output: [b.cols, a.rows],
                constants: {
                    ySize: a.cols
                },
            })(a.d, b.d)); //.setOutput([a.rows, b.cols]));
        };

        const hadProductKernel = (settings) => gpu.createKernel(function (a, b) {
            let sum = a[this.thread.y][this.thread.x] * b[this.thread.y][this.thread.x];
            return sum;
        }, settings);

        const hadProduct = (a, b) => {
            if (a.rows !== b.rows || a.cols !== b.cols) {
                throw "HADPRODUCT: sizes of mat a and b do not match";
            }
            return new Matrix(hadProductKernel({
                output: [a.cols, a.rows]
            })(a.d, b.d));
        }

        const scalarProductKernel = (settings) => gpu.createKernel(function (a, scalar) {
            let sum = a[this.thread.y][this.thread.x] * scalar;
            return sum;
        }, settings);

        const scalarProduct = (a, scalar) => {
            return new Matrix(scalarProductKernel({
                output: [a.cols, a.rows]
            })(a.d, scalar));
        }

        const sumKernel = (settings) => gpu.createKernel(function (a, b) {
            let sum = a[this.thread.y][this.thread.x] + b[this.thread.y][this.thread.x];
            return sum;
        }, settings);

        const sum = (a, b) => {
            if (a.rows !== b.rows || a.cols !== b.cols) {
                throw "SUM: sizes of mat a and b do not match";
            }
            return new Matrix(sumKernel({
                output: [a.cols, a.rows]
            })(a.d, b.d));
        }

        const subtractKernel = (settings) => gpu.createKernel(function (a, b) {
            let sum = a[this.thread.y][this.thread.x] - b[this.thread.y][this.thread.x];
            return sum;
        }, settings);

        const subtract = (a, b) => {
            if (a.rows !== b.rows || a.cols !== b.cols) {
                throw "SUBTRACT: sizes of mat a and b do not match";
            }
            return new Matrix(subtractKernel({
                output: [a.cols, a.rows]
            })(a.d, b.d));
        }

        const mapSigmoidKernel = (settings) => gpu.createKernel(function (a) {
            let sum = 1.0 / (1.0 + Math.exp(a[this.thread.y][this.thread.x]));
            return sum;
        }, settings);

        const mapSigmoid = (a) => {
            return new Matrix(mapSigmoidKernel({
                output: [a.cols, a.rows]
            })(a.d));
        }

        const mapSigmoid_PKernel = (settings) => gpu.createKernel(function (a) {
            let sum = a[this.thread.y][this.thread.x] * (1.0 - a[this.thread.y][this.thread.x]);
            return sum;
        }, settings);

        const mapSigmoid_P = (a) => {
            return new Matrix(mapSigmoid_PKernel({
                output: [a.cols, a.rows]
            })(a.d));
        }

        const mapReluKernel = (settings) => gpu.createKernel(function (a) {
            let sum = Math.max(0, a[this.thread.y][this.thread.x]);
            return sum;
        }, settings);

        const mapRelu = (a) => {
            return new Matrix(mapReluKernel({
                output: [a.cols, a.rows]
            })(a.d));
        }

        const fromArrayKernel = (settings) => gpu.createKernel(function (arr) {
            let sum = arr[this.thread.y];
            return sum;
        }, settings);

        const fromArray = (arr) => {
            return new Matrix(fromArrayKernel({
                output: [1, arr.length]
            })(arr));
        }

        const transposeKernel = (settings) => gpu.createKernel(function (a) {
            let sum = a[this.thread.x][this.thread.y];
            return sum;
        }, settings);

        const transpose = (a) => {
            return new Matrix(transposeKernel({
                output: [a.rows, a.cols]
            })(a.d));
        }

        return {
            dotProduct,
            hadProduct,
            scalarProduct,
            sum,
            subtract,
            mapSigmoid,
            mapSigmoid_P,
            mapRelu,
            fromArray,
            transpose,
        }
    }
}

module.exports = {
    Matrix
};