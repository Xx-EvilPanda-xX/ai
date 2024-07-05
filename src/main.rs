use std::os::unix::fs::FileExt;
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

    let mut network = Network::new(vec![(IMG_WIDTH * IMG_HEIGHT) as usize, 30, 20, 10], true);
    let input = vec_from_train_image(34001);
    let mut out = vec![0.0; 10];
    network.compute(&input, &mut out);
    println!("{:?}", out);
    let ideal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let now = std::time::Instant::now();
    println!("cost function (.back_propagate): {}", network.back_propagate(&input, &ideal));
    println!("Time elapsed: {}", std::time::Instant::now().duration_since(now).as_secs_f64());
    println!("cost function: {}", out.iter().zip(ideal.iter()).map(|(o, i)| o - i).sum::<f64>());

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