use std::os::windows::prelude::FileExt;

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
    let mut network = Network::new(vec![1, 30, 1], true);
    // const INPUT_RANGE: Range<f64> = 0.0..(100.0 * std::f64::consts::PI);
    let mut out = vec![0.0; 1];

    let inputs = [-0.045519648967676574, -7.212972091875587, -3.4815316413363018, -6.8493437125136625, -9.16065322245121, 0.23989636528888525, 4.470162504794457, 9.447478020077291, 3.4446477596643543, -8.50568841693265, 8.716208777492938, 9.578502174317936, -6.17182776126977, -0.01735274489565697, -4.726141056454596, 1.0485258898832015, -6.01974203544827, -5.890803507591471, -2.453447544760232, 5.461810224333753, 6.957616730202517, 0.04763532908670953, -4.673568606372145, 8.628939503074836, 0.752083226865583, 5.952080991035501, 0.15022228466554566, -7.122554434396049, 9.15554764108218, 5.934232190877001, -0.04996143555544208, -0.9008508880539381, -5.320611072846879, -3.998806145801286, 4.432209617896486, 2.322756114269403, -2.1823070830211355, -2.3390847219079136, 2.243265645385325, -7.349546336149766, 5.613354030432527, -2.26772106364566, 8.577408706408566, 9.470288199084646, -0.9581321722785852, -0.33077981053698835, 4.014257053979682, 6.140146563374515, -9.159910827882296, -9.605452510771997, ];
    let ideals = [-0.04551866649005353, -3.8924227579535793, -3.0583031669400733, -3.959999927768349, -3.008891931789224, 0.23975257761269636, 3.59610486050566, 2.812330299286503, 3.034400577113938, -3.3982820986137745, 3.282583709463695, 2.717665986325871, -3.9984500372308203, -0.017352690466337424, -3.7007589709062434, 1.0365591999447028, -3.991327840733136, -3.980769993251015, -2.3024791236435354, 3.9159637849664466, 3.9432773520126316, 0.04763420315202719, -3.6804884123647255, 3.331667430619319, 0.747659800637203, 3.9863040644001653, 0.15018697438012446, -3.9122551219162243, 3.0122535914852056, 3.984788616208298, -0.04996013649276612, -0.8932548689900626, -3.884739192594928, -3.365238747147432, 3.5793238667900202, 2.1944006132702607, -2.07564486664357, -2.208034393095726, 2.1275109180192824, -3.858699110506511, 3.9440466952346136, -2.1481803924947607, 3.359907224674211, 2.7960641545444496, -0.9489961101166497, -0.33040293560343714, 3.3735656620328363, 3.9974427622455013, -3.0093810501520015, -2.6978295660935574, ];
    // let inputs = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    // let ideals = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0];
    assert_eq!(inputs.len(), ideals.len());

    // let input = vec_from_train_image(34001);

    for _ in 0..15 {
        let mut sum_costs = 0.0;

        for i in 1..1000 {
            let idx = rng.gen_range(0..inputs.len());

            let input = [inputs[idx]];
            let ideal = [ideals[idx]];
            let cost = network.back_propagate(&input, &ideal).sqrt();

            network.compute(&input, &mut out);

            sum_costs += cost;
            let average_cost = sum_costs / i as f64;
            println!("propagate! cost = {cost}, out = {out:?}, input = {input:?}, ideal = {ideal:?}, average cost = {average_cost}");
        }
    }

    let mut input = -10.0;

    while input < 10.0 {
        network.compute(&[input], &mut out);
        println!("({input}, {})", out[0]);
        input += 1.0;
    }

    println!("Incoming weights for layer 1:");
    for weights in network.get_weights(0) {
        for weight in weights {
            println!("{weight}");
        }
    }

    println!("\nBiases for layer 1:");
    for bias in network.get_biases(0) {
        println!("{bias}");
    }

    println!("Incoming weights for layer 2:");
    for weights in network.get_weights(1) {
        for weight in weights {
            println!("{weight}");
        }
    }

    println!("\nBiases for layer 2:");
    for bias in network.get_biases(1) {
        println!("{bias}");
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
    stream.seek_read(&mut buf, BEGIN_DATA_OFFSET + (idx * IMG_WIDTH as u64 * IMG_HEIGHT as u64)).expect("Failed to read training image");
    image::ImageBuffer::from_vec(IMG_WIDTH, IMG_HEIGHT, buf).expect("Failed to contruct training image")
}

fn vec_from_train_image(idx: u64) -> Vec<f64> {
    assert!(idx < NUM_IMGS);

    let stream = std::fs::File::open(TRAIN_IMG_PATH).expect("Failed to open training data from file");
    let mut buf = vec![0; (IMG_WIDTH * IMG_HEIGHT) as usize];
    stream.seek_read(&mut buf, BEGIN_DATA_OFFSET + (idx * IMG_WIDTH as u64 * IMG_HEIGHT as u64)).expect("Failed to read training image");

    buf.iter().map(|x| *x as f64).collect()
}