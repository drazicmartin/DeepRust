use rand::distributions::{Distribution, Uniform};
mod core;
mod digits_recognition;

use core::network;
use core::loss;

// fn train(network: &mut network::Network, input_size: usize, iter: i32){
//     let mut rng = rand::thread_rng();
//     let dist = Uniform::from(0f32..=1f32);
//     for i in 0..iter {
//         let input: Vec<f32> = dist.sample_iter(&mut rng).take(input_size).collect();
//         let a: f32 = network.forward(input.clone())[0];

//         println!("output : {}", a);

//         let y_hat = 1f32;

//         let loss = (loss::LOSS_BCE.forward)(a, y_hat);
//         let gradient = (loss::LOSS_BCE.derivate)(a, y_hat);

//         network.backward(gradient);
//         network.update(0.1f32);

//         // if i%10 == 0 {
//         //     println!("{}", loss);
//         // }
//     }
// }

fn main() {
    // let input_size : usize = 10;
    // let mut network : network::Network = network::build_model(input_size);
    // train(&mut network, input_size, 10);


    digits_recognition::main();
}
