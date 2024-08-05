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

        let inputs = [6.306521523288275, 7.36313802853056, -9.286330273055023, 9.246205196853587, 9.285800858557486, 5.810517195377951, 3.299994953510218, -7.5783439370894, -5.624764540286575, 1.3404423484708126, -0.6390801603242799, 3.690754640628313, -8.132233081796647, -9.42290484318903, -5.08718412970691, 1.311896186872815, 5.355546263545037, 9.342437847854761, -5.629682422532625, -7.720555529938307, 9.012871652547307, -2.5089358575189635, 3.654139594277508, 2.2770870168461457, 1.8061916533746256, 1.0871596010739193, -5.795437860162771, 4.7902681884153875, -6.529989963194742, 5.45731963604489, -8.902088315125232, -8.439963512210866, -8.310720148685146, 5.449323536195067, 0.046723166632887114, -5.97279636629529, -1.0245115075267677, -0.20257926895385836, -0.43078390204731676, 5.8645821547232835, 0.4613641829909021, 8.267999706032704, 0.7589936942991358, 9.106984534727086, 2.896583920317717, -1.3137683670851388, -2.6891518383444346, -3.8923121141574546, 2.195467804016795, -0.7395375749214281, ];
        let ideals = [3.9772213723698266, 5.421580162719291, 8.623592994025818, 8.549231054232228, 8.622609758478696, 3.376211007778285, 1.0889966693192907, 5.743129682881967, 3.163797613366525, 0.17967856895739473, 0.04084234513201074, 1.3621669817319428, 6.613321489666779, 8.879113568379529, 2.587944236954185, 0.17210716051314318, 2.86818757809712, 8.72811449410291, 3.16933241785728, 5.960697769086097, 8.123185542529082, 0.6294759137144417, 1.335273617446659, 0.5185125282289279, 0.3262328288720164, 0.11819159982072033, 3.358709999100804, 2.2946669316944437, 4.264076891942407, 2.9782337609961127, 7.924717637028918, 7.123298408745079, 6.906806938976125, 2.9695127002129507, 0.0002183054300204536, 3.5674296433230217, 0.10496238290547702, 0.004103836020987968, 0.01855747702631122, 3.4393323849498794, 0.021285690934686262, 6.835981913895688, 0.057607142798585, 8.293716731575833, 0.8390198407443152, 0.1725987322353552, 0.7231537609671252, 1.5150093594016873, 0.4820078878474329, 0.054691582472066694, ];
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
                let cost = network.back_propagate(&input, &ideal);

                network.compute(&input, &mut out);

                sum_costs += cost;
                let average_cost = sum_costs / i as f64;
                println!("propagate! cost = {cost}, out = {out:?}, input = {input:?}, ideal = {ideal:?}, average cost = {average_cost}");
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