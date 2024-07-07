use rand::Rng;
use std::ops::Range;
use crate::function::{Symbol, Function, Variable};

#[derive(Clone, Debug)]
pub struct Network {
    // input: IOLayer,
    hidden: Vec<Layer>,
    dims: Vec<usize>,
}

// Outer vector - nodes, inner vector - incoming weights/biases
#[derive(Clone, Debug)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    weight_indices: Vec<Vec<usize>>,
    bias_indices: Vec<usize>,
    num_weights: usize,
    num_biases: usize,
}

// Input layer and output layer only
// struct IOLayer {
//     nodes: Vec<f64>
// }

const DEFAULT_WEIGHT: f64 = 1.0;
const DEFAULT_BIAS: f64 = 0.0;

const WEIGHT_RANGE: Range<f64> = -1.0..1.0;
const BIAS_RANGE: Range<f64> = -1.0..1.0;

impl Network {

    // length of dims = num layers, elements of dims = nodes in each layer
    pub fn new(dims: Vec<usize>, init_rand: bool) -> Self {
        assert_ne!(dims.len(), 0);
        let mut new = Self { hidden: Vec::new(), dims: dims.clone() };
        let mut rng = rand::thread_rng();

        let mut var_index = 0;

        let mut increment = |vec: &mut Vec<usize>| {
            vec.push(var_index);
            var_index += 1;
        };

        // iter layers
        for (i, num_nodes) in dims.iter().skip(1).enumerate() {
            new.hidden.push(
                Layer { 
                    weights: Vec::new(),
                    biases: Vec::new(),
                    weight_indices: Vec::new(),
                    bias_indices: Vec::new(),
                    num_weights: 0,
                    num_biases: 0,
                }
            );

            // iter nodes
            for j in 0..*num_nodes {
                if init_rand {
                    new.hidden[i].biases.push(rng.gen_range(BIAS_RANGE));
                } else {
                    new.hidden[i].biases.push(DEFAULT_BIAS);
                }
                increment(&mut new.hidden[i].bias_indices);
                new.hidden[i].num_biases += 1;
    
                new.hidden[i].weights.push(Vec::new());
                new.hidden[i].weight_indices.push(Vec::new());

                // iter weights/biases
                for _ in 0..dims[i] {
                    if init_rand {
                        new.hidden[i].weights[j].push(rng.gen_range(WEIGHT_RANGE));
                    } else {
                        new.hidden[i].weights[j].push(DEFAULT_WEIGHT);
                    }
                    increment(&mut new.hidden[i].weight_indices[j]);
                    new.hidden[i].num_weights += 1;
                }
            }
        }

        new
    }

    pub fn compute(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), self.dims[0]);
        assert_eq!(output.len(), self.dims[self.dims.len() - 1]);

        let mut last = input.to_vec();
        let mut working = vec![0.0; self.dims[1]];

        for (layer_idx, layer) in self.hidden.iter().enumerate() {
            for (i, (node_weights, node_bias)) in layer.weights.iter().zip(layer.biases.iter()).enumerate() {
                for (j, weight) in node_weights.iter().enumerate() {
                    working[i] += last[j] * weight;
                }
                
                // add bias/apply activation function
                if layer_idx != self.hidden.len() - 1 {
                    working[i] = sig(working[i] + node_bias);
                } else {
                    working[i] += node_bias;
                }
            }
            
            std::mem::swap(&mut working, &mut last);

            if layer_idx < self.dims.len() - 2 {
                working = vec![0.0; self.dims[layer_idx + 2]]
            }
        }

        output.copy_from_slice(&last);
    }

    fn gen_derivative_input(&self) -> Vec<f64> {
        let flat_weight: Vec<_> = self.hidden.iter().map(|x| x.weights.iter().flatten()).flatten().collect();
        let flat_bias: Vec<_> = self.hidden.iter().map(|x| x.biases.iter()).flatten().collect();
        let mut function_input = vec![0.0; flat_weight.len() + flat_bias.len()];

        for layer in self.hidden.iter() {
            for (weight_node, weight_index_node) in layer.weights.iter().zip(layer.weight_indices.iter()) {
                for (weight, weight_index) in weight_node.iter().zip(weight_index_node.iter()) {
                    function_input[*weight_index] = *weight;
                }
            }

            for (bias, bias_index) in layer.biases.iter().zip(layer.bias_indices.iter()) {
                function_input[*bias_index] = *bias;
            }
        }

        function_input
    }

    pub fn back_propagate(&mut self, input: &[f64], ideal: &[f64]) -> f64 {
        let function_input = self.gen_derivative_input();
    
        let cost = self.cost(input, ideal);
        let cost_closure = cost.to_closure();
        let y = cost_closure(&function_input);

        let copy = self.clone();

        // back propagate
        for layer in self.hidden.iter_mut() {
            for (weight_node, weight_index_node) in layer.weights.iter_mut().zip(layer.weight_indices.iter()) {
                for (weight, weight_index) in weight_node.iter_mut().zip(weight_index_node.iter()) {
                    let derivative = cost.derivative(*weight_index);
                    let derivative_closure = derivative.to_closure();
                    let d_y = derivative_closure(&function_input);

                    *weight -= d_y / y;
                }
            }

            for (bias, bias_index) in layer.biases.iter_mut().zip(layer.bias_indices.iter()) {
                let derivative = cost.derivative(*bias_index);
                let derivative_closure = derivative.to_closure();
                let d_y = derivative_closure(&function_input);

                *bias -= d_y / y;
            }
        }

        let function_input = self.gen_derivative_input();

        if cost_closure(&function_input) > y {
            *self = copy;
        }

        y
    }

    // pub fn back_propagate(&mut self, input: &[f64], ideal: &[f64]) -> f64 {
    //     let flat_weight_indices: Vec<_> = self.hidden.iter().map(|x| x.weight_indices.iter().flatten()).flatten().collect();
    //     let flat_bias_indices: Vec<_> = self.hidden.iter().map(|x| x.bias_indices.iter()).flatten().collect();
    //     let flat_weight: Vec<_> = self.hidden.iter().map(|x| x.weights.iter().flatten()).flatten().collect();
    //     let flat_bias: Vec<_> = self.hidden.iter().map(|x| x.biases.iter()).flatten().collect();

    //     let mut function_input = vec![0.0; flat_weight.len() + flat_bias.len()];

    //     for (i, v) in function_input.iter_mut().enumerate() {
    //         match flat_weight_indices.iter().position(|p| **p == i) {
    //             Some(i) => *v = *flat_weight[i],
    //             None => {
    //                 match flat_bias_indices.iter().position(|p| **p == i) {
    //                     Some(i) => *v = *flat_bias[i],
    //                     None => println!("Error: no such index"),
    //                 }
    //             }
    //         }
    //     }

    //     let now = std::time::Instant::now();
    //     let cost = self.cost(input, ideal);
    //     println!(".cost elapsed: {}", std::time::Instant::now().duration_since(now).as_secs_f64());
    //     let now = std::time::Instant::now();
    //     let closure = cost.to_closure();
    //     println!(".to_closure elapsed: {}", std::time::Instant::now().duration_since(now).as_secs_f64());
    //     let now = std::time::Instant::now();
    //     let x = closure(&function_input);
    //     println!("execution elapsed: {}", std::time::Instant::now().duration_since(now).as_secs_f64());
    //     x
    // }

    fn node_function(&self, input: &[f64], layer: usize, node: usize, last_layer: bool) -> Symbol {
        // println!("node_function - layer: {}, node: {}", layer, node);
        if layer == 0 {
            return Symbol::Func(Function::Const(input[node]));
        }

        let mut sum = Vec::new();

        let bias_index = self.hidden[layer - 1].bias_indices[node];

        for (i, weight_index) in self.hidden[layer - 1].weight_indices[node].iter().enumerate() {
            sum.push(
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Var(Variable { index: *weight_index }),
                    self.node_function(input, layer - 1, i, false)
                ))))
            )
        }

        sum.push(Symbol::Var(Variable { index: bias_index }));

        if last_layer {
            Symbol::Func(Function::Add(sum))
        } else {
            Symbol::Func(Function::Sig(Box::new(
                Symbol::Func(Function::Add(sum))
            )))
        }
    }

    fn cost(&self, input: &[f64], ideal: &[f64]) -> Symbol {
        assert_eq!(ideal.len(), self.dims[self.dims.len() - 1]);
        assert_eq!(input.len(), self.dims[0]);

        let mut out_nodes_diff = Vec::new();

        for (i, node) in ideal.iter().enumerate() {
            out_nodes_diff.push(
                Symbol::Func(Function::Mul(Box::new((
                    Symbol::Func(Function::Add( // TODOOOOOO: THIS NOT A PROPER COST FUNCTION. CHANGE TO BE DISTANCE IN HIGH DIMENSIONAL SPACE.
                        vec![
                            Symbol::Func(Function::Const(-(*node))),
                            self.node_function(input, self.dims.len() - 1, i, true)
                        ]
                    )),
                    Symbol::Func(Function::Add( // TODOOOOOO: THIS NOT A PROPER COST FUNCTION. CHANGE TO BE DISTANCE IN HIGH DIMENSIONAL SPACE.
                        vec![
                            Symbol::Func(Function::Const(-(*node))),
                            self.node_function(input, self.dims.len() - 1, i, true)
                        ]
                    ))
                ))))
            )
        }

        Symbol::Func(Function::Add(out_nodes_diff))
    }
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
