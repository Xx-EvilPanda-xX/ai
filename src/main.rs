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
    let args: Vec<_> = std::env::args().collect();

    if args.len() > 1 {
        test(&Network::new_from_save(&args[1]));
    } else {
        let mut network = Network::new(vec![1, 30, 1], true);

        // const INPUT_RANGE: Range<f64> = 0.0..(100.0 * std::f64::consts::PI);
        let mut out = vec![0.0; 1];

        let inputs = [-4.426113629537312, 12.749157123671122, -5.692361733326559, -0.29314182575211056, -1.761326324645637, 14.205128417949474, -8.627043797318208, 4.0641528820486315, -11.025780836526149, 6.092242636206503, -3.315434268946424, -6.8078519250341, -7.81216795621261, 13.058012836036255, -5.530070767289912, 8.498475585554448, -10.739201350293401, -6.460866461044187, 10.992794299673761, -6.692941398751142, 2.7877163479796288, -1.629106521750101, 3.8270396546977707, -14.448604218317321, -5.807316906571039, -0.20957833032420048, 9.2313839787721, 9.734219342123417, -5.047230687147302, 10.630510134422867, 4.059734908481122, -9.64080308813138, 10.793927567466326, -7.814747080672804, -5.466679815093855, -0.9209078734216458, -11.828862977418042, 10.119649613064272, -3.4069505835492464, 14.238464440971939, 8.204504913498354, 3.486118973219419, -12.059629725080438, 1.8076673838223947, -12.248071142020464, 3.9075665723687294, -2.513880435580564, 10.65630602894846, 5.186852688047459, 2.00053412712969, ];
        let ideals = [1.959048186157596, 16.254100736405412, 3.2402982103040543, 0.008593213000528076, 0.3102270421889708, 20.178567337043575, 7.442588468084658, 1.6517338648664197, 12.156784305510726, 3.711542033841236, 1.099210439170431, 4.6346847833190505, 6.102996817607511, 17.05116992260876, 3.0581682691234433, 7.222408727826503, 11.53304456421436, 4.174279542744563, 12.084152651493994, 4.479546456711689, 0.7771362436792878, 0.2653988059208713, 1.4646232518629232, 20.876216385757708, 3.372492965334582, 0.004392307654147969, 8.521845016353023, 9.475502620056965, 2.5474537609281422, 11.300774571806727, 1.6481447527140225, 9.294508418412354, 11.650887233170952, 6.1070271934884115, 2.9884588200754583, 0.0848071311329978, 13.992199933853124, 10.240730829119187, 1.160731227874655, 20.273386963682235, 6.731390087561865, 1.2153025495440417, 14.54346691060437, 0.3267661370535301, 15.00152466999945, 1.52690765174935, 0.6319594844394726, 11.35568581826033, 2.6903440807505152, 0.40021367938105507, ];
        // let inputs = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        // let ideals = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0];
        assert_eq!(inputs.len(), ideals.len());

        // let input = vec_from_train_image(34001);
        let mut rng = rand::thread_rng();

        for _ in 0..15 {
            let mut sum_costs = 0.0;

            for i in 1..1000 {
                let idx = rng.gen_range(0..inputs.len());

                let input = [inputs[idx]];
                let ideal = [ideals[idx]];
                let (pre_cost, post_cost) = network.back_propagate(&input, &ideal);

                network.compute(&input, &mut out);

                sum_costs += pre_cost;
                let average_cost = sum_costs / i as f64;
                println!("propagate! pre-cost = {pre_cost}, post_cost = {post_cost}, out = {out:?}, input = {input:?}, ideal = {ideal:?}, average cost = {average_cost}");
            }
        }

        test(&network);
        network.save("network.dat");
        network::save_network_text("network.txt", &network).expect("Failed to save network to text");
    };
}

fn test(network: &Network) {
    // assert_eq!(network.get_dims(), &[1, 30, 1]);
    let mut out = vec![0.0; 1];
    let mut input = -10.0;

    while input < 10.0 {
        network.compute(&[input], &mut out);
        println!("({input}, {})", out[0]);
        input += 1.0;
    }
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