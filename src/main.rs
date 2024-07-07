use std::os::unix::fs::FileExt;
use rand::Rng;
use network::Network;

mod function;
mod network;

const TRAIN_IMG_PATH: &str = "./MNIST_ORG/train-images.idx3-ubyte";
const TRAIN_LABEL_PATH: &str = "./MNIST_ORG/train-labels.idx1-ubyte";
const NUM_IMGS: u64 = 60000;
const IMG_WIDTH: u32 = 28;
const IMG_HEIGHT: u32 = 28;
const BEGIN_DATA_OFFSET: u64 = 16;

fn main() {
    // let images: Vec<_> = (0..20).map(|i| image_from_set(i)).collect(); 
    // for (i, image) in images.iter().enumerate() {
    //     image.save(format!("data{}.png", i).as_str()).expect("Failed to save training image");
    // }

    let mut rng = rand::thread_rng();
    let mut network = Network::new(vec![1, 20, 20, 1], true);

    // let input = vec_from_train_image(34001);

    for _ in 0..100 {
        let mut sum_costs = 0.0;

        for i in 1..100 {
            let input = [rng.gen_range(0.0..(100.0 * std::f64::consts::PI))];
            let ideal = [(input[0] / 100.0).sin() * 100.0];
            let mut out = vec![0.0; 1];
            network.compute(&input, &mut out);
            let cost = network.back_propagate(&input, &ideal);
            sum_costs += cost;
            let average_cost = sum_costs / i as f64;
            println!("propagate! cost = {cost}, out = {out:?}, input = {input:?}, ideal = {ideal:?}, average cost = {average_cost}");
        }
    }

    let mut out = vec![0.0; 1];
    network.compute(&[0.0], &mut out);
    println!("out: {:?}", out);

    let image = image_from_set(34001);
    image.save("34001.png").expect("Failed to save image");
}   

fn image_from_set(idx: u64) -> image::ImageBuffer<image::Luma<u8>, Vec<u8>> {
    assert!(idx < NUM_IMGS);
    let stream = std::fs::File::open(TRAIN_IMG_PATH).expect("Failed to open training data from file");
    let mut buf = vec![0; (IMG_WIDTH * IMG_HEIGHT) as usize];
    stream.read_at(&mut buf, BEGIN_DATA_OFFSET + (idx * IMG_WIDTH as u64 * IMG_HEIGHT as u64)).expect("Failed to read training image");
    image::ImageBuffer::from_vec(IMG_WIDTH, IMG_HEIGHT, buf).expect("Failed to contruct training image")
}

fn vec_from_train_image(idx: u64) -> Vec<f64> {
    assert!(idx < NUM_IMGS);

    let stream = std::fs::File::open(TRAIN_IMG_PATH).expect("Failed to open training data from file");
    let mut buf = vec![0; (IMG_WIDTH * IMG_HEIGHT) as usize];
    stream.read_at(&mut buf, BEGIN_DATA_OFFSET + (idx * IMG_WIDTH as u64 * IMG_HEIGHT as u64)).expect("Failed to read training image");

    buf.iter().map(|x| *x as f64).collect()
}