use std::ptr::null;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;

struct Function {
    derivate: fn(f32) -> f32,
    forward: fn(f32) -> f32,
}

struct LossFunction {
    derivate: fn(f32, f32) -> f32,
    forward: fn(f32, f32) -> f32,
}

fn sigmoid(x : f32) -> f32{
    1f32/(1f32 + (-x).exp())
}

const EPSILON: f32 = 1e-10;

const SIGMOID: Function =  Function {
    derivate: |x: f32| -> f32 {
        sigmoid(x)*(1f32-sigmoid(x))
    },
    forward: |x: f32| -> f32 {
        sigmoid(x)
    },
};

const LOSS_BCE: LossFunction =  LossFunction {
    derivate: |a: f32, y_hat: f32| -> f32 {
        match a {
            a if a == 0f32 => -100f32,
            _ => -y_hat/a + (1f32-y_hat)/(1f32-a + EPSILON)
        }
    },
    forward: |a: f32, y_hat: f32| -> f32 {
        - y_hat * a.log10() - (1f32-y_hat)*(1f32-a + EPSILON).log10()
    },
};

// Node will be the basic of Perceptron
struct Node {
    auto_grad: bool,
    weights: Vec<f32>,
    weight_gradients: Vec<f32>,
    bias: f32,
    bias_gradient: f32,
    activation_function: Function,
    saved_input: Vec<f32>,
    saved_z: f32,
    input_size: usize,
}

impl Node {
    fn new(input_size: usize) -> Self {
        Node {
            auto_grad: true,
            activation_function: SIGMOID,
            weights: vec![0f32; input_size],
            weight_gradients: vec![0f32; input_size],
            bias_gradient: 0f32,
            bias: 0f32,
            input_size: input_size,
            saved_input: vec![0f32; input_size],
            saved_z: 0f32,
        }
    }

    fn set_autograd(mut self, value: bool){
        self.auto_grad = value
    }
    fn disable_auto_grad(mut self){
        self.set_autograd(false)
    }
    fn enable_auto_grad(mut self){
        self.set_autograd(true)
    }

    fn init_weight(&mut self) {
        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-100f32..=100f32);
        
        for w in &mut self.weights{
            *w = dist.sample(&mut rng);
        }
        self.bias = dist.sample(&mut rng);
    }

    fn forward(&mut self, input: Vec<f32>) -> f32 {
        let mut weighted_sum: f32 = 0f32;
        // assert!(input.len() == self.weights.len());
        
        for (x, w) in input.iter().zip(self.weights.iter()).collect::<Vec<_>>() {
            weighted_sum += x*w
        }
        self.saved_input = input;
        self.saved_z = weighted_sum;
        weighted_sum += self.bias;

        (self.activation_function.forward)(weighted_sum)
    }

    fn backward(&mut self, accumulated_gradient: f32) -> Vec<f32>{
        self.bias_gradient = accumulated_gradient * (self.activation_function.derivate)(self.saved_z);
        let mut accumulated_gradients: Vec<f32> = vec![self.bias_gradient; self.input_size];
        
        for idx in 0..self.input_size {
            self.weight_gradients[idx] = self.bias_gradient * self.saved_input[idx];

            accumulated_gradients[idx] *= self.weights[idx];
        }

        accumulated_gradients
    }

    fn update(&mut self, learning_rate: f32) {
        self.bias -= learning_rate * self.bias_gradient;
        for idx in 0..self.input_size {
            self.weights[idx] -= learning_rate * self.weight_gradients[idx];
        }
    }
}

// Layer wll be the assemble of multiple Nodes
struct Layer {
    pub nodes: Vec<Node>,
    input_size: usize,
    size: usize,
}

impl Layer {
    fn new(input_size: usize, size: usize) -> Self {
        Layer {
            nodes: Layer::get_default_nodes(input_size, size),
            input_size,
            size,
        }
    }

    fn set_autograd(mut self, value: bool){
        for node in self.nodes{
            node.set_autograd(value);
        }
    }
    fn disable_auto_grad(mut self){
        self.set_autograd(false);
    }
    fn enable_auto_grad(mut self){
        self.set_autograd(true);
    }

    fn get_default_nodes(input_size: usize, size: usize) -> Vec<Node> {
        let mut nodes: Vec<Node> = Vec::new();
        for _idx in 0..size {
            let mut node = Node::new(input_size);
            node.init_weight();
            nodes.push(
                node
            );
        }
        nodes
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::new();
        for node in &mut self.nodes {
            output.push(node.forward(input.clone()));
        }
        output
    }

    fn backward(&mut self, accumulated_gradients: Vec<f32>) -> Vec<f32> {
        let size = accumulated_gradients.len();
        let mut average_gradients: Vec<f32> = vec![0f32; size];
        for idx in 0..self.nodes.len() {
            println!(" layer nodes : {}", self.nodes.len());
            println!(" layer accumulated grad : {}", accumulated_gradients[idx]);
            let node_output = self.nodes[idx].backward(accumulated_gradients[idx]);
            average_gradients = average_gradients.iter().zip(&node_output).map(|(a, b)| a + b).collect();
        }
        average_gradients = average_gradients.iter().map(|a| a / size as f32).collect();

        average_gradients
    }

    fn update(&mut self, learning_rate: f32){
        for idx in 0..self.nodes.len() {
            self.nodes[idx].update(learning_rate);
        }
    }
}

// Network will be the assemble of multiple Layers
struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    fn new() -> Self {
        Network {
            layers: Vec::new(),
        }
    }

    fn set_autograd(mut self, value: bool){
        for layer in self.layers{
            layer.set_autograd(value);
        }
    }
    fn disable_auto_grad(mut self){
        self.set_autograd(false);
    }
    fn enable_auto_grad(mut self){
        self.set_autograd(true);
    }

    fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = input;
        for layer in &mut self.layers {
            output = layer.forward(output);
        }
        output
    }

    fn backward(&mut self, loss_gradient: f32) {
        if let Some(last_layer) = self.layers.last() {
            let size_last_layer = last_layer.nodes.len();
            println!("grad {:?}", loss_gradient);
            let mut accumulated_gradients = vec![loss_gradient; size_last_layer];


            for idx in (0..self.layers.len()).rev(){
                accumulated_gradients = self.layers[idx].backward(accumulated_gradients);
            }
        } else {
            println!("Network is empty");
        }
    }

    fn update(&mut self, learning_rate: f32) {
        for idx in (0..self.layers.len()).rev(){
            self.layers[idx].update(learning_rate);
        }
    }
}

fn build_model(input_size: usize) -> Network {
    let mut mlp = Network::new();

    let layer_one = Layer::new(input_size, 20);
    // print!("{:?}", layer_one.nodes[0].weights);

    mlp.add_layer(
        layer_one
    );
    mlp.add_layer(
        Layer::new(20, 30)
    );
    mlp.add_layer(
        Layer::new(30, 40)
    );
    mlp.add_layer(
        Layer::new(40, 1)
    );

    mlp
}

fn train(network: &mut Network, input_size: usize, iter: i32){
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(0f32..=1f32);
    for i in 0..iter {
        let input: Vec<f32> = dist.sample_iter(&mut rng).take(input_size).collect();
        let a: f32 = network.forward(input.clone())[0];

        println!("output : {}", a);

        let y_hat = 1f32;

        let loss = (LOSS_BCE.forward)(a, y_hat);
        let gradient = (LOSS_BCE.derivate)(a, y_hat);
        network.backward(gradient);
        network.update(0.1);

        // if i%10 == 0 {
        //     println!("{}", loss);
        // }
    }
}

fn main() {
    let input_size : usize = 10;
    let mut network : Network = build_model(input_size);
    train(&mut network, input_size, 100);

}
