use csv::StringRecord;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use crate::core::network;
use crate::core::loss;


pub(crate) fn build_model(input_size: usize) -> network::Network {
    let mut mlp = network::Network::new();

    mlp.add_layer(
        network::Layer::new_with_activation(input_size, 20, network::RELU)
    );
    // mlp.add_layer(
    //     network::Layer::new_with_activation(20, 30, network::RELU)
    // );
    // mlp.add_layer(
    //     network::Layer::new_with_activation(30, 40, network::RELU)
    // );
    // mlp.add_layer(
    //     network::Layer::new_with_activation(40, 70, network::RELU)
    // );
    mlp.add_layer(
        network::Layer::new_with_activation(20, 10, network::LINEAR)
    );

    mlp
}

fn read_csv_file<P: AsRef<Path>>(file_path: P) -> Result<Vec<(Vec<f32>, usize)>, Box<dyn Error>> {
    // Open the file in read-only mode
    let file = File::open(file_path)?;
    
    // Create a CSV Reader from the file
    let mut rdr = csv::Reader::from_reader(file);

    // Vector to hold records
    let mut records: Vec<(Vec<f32>, usize)> = Vec::new();

    // Iterate over each record
    for result in rdr.records() {
        // Extract the record
        let record: StringRecord = result?;
        let parsed_label: usize = record[0].parse()?;
        let parsed_image: Vec<f32> = record.iter()
            .skip(1) // Skip the first element
            .map(|field| field.parse::<f32>().unwrap()) // Note: Handle parsing errors as needed
            .collect();

            records.push((parsed_image, parsed_label));
    }

    Ok(records)
}

fn train(network: &mut network::Network, epoch: i32){
    for i in 0..epoch {
        match read_csv_file("data/train.csv") {
            Ok(records) => {

                let mut iter = 1;
                // Process each record as needed
                for (image, label) in &records {
                    let output = network.forward(image.to_vec());
                    let loss = (loss::LOSS_CROSS_ENTROPY.forward)(output.clone(), *label);
                    let gradients = (loss::LOSS_CROSS_ENTROPY.derivate)(output.clone(), *label);

                    println!("output: {:?}", output);
                    println!("loss: {:?}", loss[0]);
                    println!("gradients: {:?}", gradients);

                    network.backward(gradients);
                    network.update(0.1f32);
                    
                    // if i%1000 == 0 {
                    //     println!("Loss : {}", loss[0]);
                    // }

                    iter += 1;
                }
            },
            Err(err) => {
                eprintln!("Error reading CSV file: {}", err);
            }
        }
    }
}

pub(crate) fn main(){
    let mut model: network::Network = build_model(28*28);
    train(&mut model, 1);
}