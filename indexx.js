var GPU = require("gpu.js").GPU;
var Matrix = require("./matrix");

function genMatrix(xs, ys) {
    var out = new Array(xs);
    for (var x = 0; x < xs; x++) {
        out[x] = new Array(ys);
        for (var y = 0; y < ys; y++) {
            out[x][y] = 1;
        }
    }
    return out;
}

var gpu = new GPU();
var gpu2 = new GPU();

function callback(a, b) {
    let sum = 0;
    for (let i = 0; i < 5; i++) {
        sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
}

var multiplyMatrix = gpu.createKernel(callback).setOutput([5, 5]);
multiplyMatrix = gpu2.createKernel(callback).setOutput([5, 5]);

var a = genMatrix(5, 5);
var b = genMatrix(5, 5);

console.log(a, "\n\n", b)
var c = multiplyMatrix(a, b);
console.log(c);