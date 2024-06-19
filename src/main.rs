use rand::distributions::{Distribution, Uniform};
use rand::Rng;

struct Function {
    derivate: fn(f32) -> f32,
    forward: fn(f32) -> f32,
}

fn sigmoid(x : f32) -> f32{
    1f32/(1f32 + (-x).exp())
}

const SIGMOID: Function =  Function {
    derivate: |x: f32| -> f32 {
        sigmoid(x)*(1f32-sigmoid(x))
    },
    forward: |x: f32| -> f32 {
        sigmoid(x)
    },
};

// Node will be the basic of Perceptron
struct Node {
    auto_grad: bool,
    gradients: Vec<f32>,
    weights: Vec<f32>,
    bias: f32,
    activation_function: Function,
    input_size: usize,
}

impl Node {
    fn new(input_size: usize) -> Self {
        Node {
            auto_grad: true,
            gradients: vec![0f32; input_size],
            activation_function: SIGMOID,
            bias: 0f32,
            input_size: input_size,
            weights: vec![0f32; input_size],
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

    fn forward(mut self, input: Vec<f32>) -> f32 {
        let mut weighted_sum: f32 = 0f32;
        // assert!(input.len() == self.weights.len());

        for (x, w) in input.iter().zip(self.weights.iter()).collect::<Vec<_>>() {
            weighted_sum += x*w
        }
        weighted_sum += self.bias;

        for idx in 0..self.input_size {
            self.gradients[idx] = input[idx]*((self.activation_function.derivate)(weighted_sum))
        }

        (self.activation_function.forward)(weighted_sum)
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

    fn forward(self, input: Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = Vec::new();
        for node in self.nodes {
            output.push(node.forward(input.clone()));
        }
        output
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

    fn forward(self, input: Vec<f32>) -> Vec<f32> {
        let mut output: Vec<f32> = input;
        for layer in self.layers {
            output = layer.forward(output);
        }
        output
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
        Layer::new(40, 10)
    );

    mlp
}


fn main() {
    let input_size : usize = 10;
    let network : Network = build_model(input_size);
    let input = vec![1f32; input_size];
    let output: Vec<f32> = network.forward(input.clone());

    println!("Input  : {:?}", input);
    println!("Output : {:?}", output);
}
