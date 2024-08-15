use rand::Rng;
use std::ops::Range;
use std::{fs::File, io::Write};
use crate::function::{Symbol, Function, Variable};
use savefile::prelude::*;
use savefile_derive::Savefile;

#[derive(Clone, Debug, Savefile)]
pub struct Network {
    hidden: Vec<Layer>,
    dims: Vec<usize>,
}

// Outer vector - nodes, inner vector - incoming weights/biases
#[derive(Clone, Debug, Savefile)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    weight_indices: Vec<Vec<usize>>,
    bias_indices: Vec<usize>,
    num_weights: usize,
    num_biases: usize,
}

// All nodes of the network given a certain input
struct NetworkState {
    layers: Vec<LayerState>
}

struct LayerState {
    nodes: Vec<f64>
}


// ID structs, NodeID works for both nodes and biases
#[derive(Clone, Copy, Debug)]
struct NodeID {
    layer: usize,
    node: usize,
}

// ALWAYS refers to layer to which the weight in question is incoming
#[derive(Clone, Copy, Debug)]
struct WeightID {
    layer: usize,
    node: usize,
    weight: usize,
}

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

    pub fn new_from_save(path: &str) -> Self {
        load_file(path, 0).expect("Failed to load network")
    }

    pub fn get_weights(&self, layer: usize) -> &Vec<Vec<f64>> {
        &self.hidden[layer].weights
    }

    pub fn get_biases(&self, layer: usize) -> &[f64] {
        &self.hidden[layer].biases
    }

    pub fn get_dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn save(&self, path: &str) {
        save_file(path, 0, self).expect("Failed to save network");
    }

    pub fn compute(&self, input: &[f64], output: &mut [f64]) {
        self.compute_internal(input, Some(output), None);
    }

    fn compute_internal(&self, input: &[f64], output: Option<&mut [f64]>, mut network_state: Option<&mut NetworkState>) {
        assert_eq!(input.len(), self.dims[0]);

        let mut last = input.to_vec();
        let mut working = vec![0.0; self.dims[1]];

        for (layer_idx, layer) in self.hidden.iter().enumerate() {
            if let Some(ref mut network_state) = network_state {
                network_state.layers.push(LayerState { nodes: last.clone() });
            }

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

        if let Some(ref mut network_state) = network_state {
            network_state.layers.push(LayerState { nodes: last.clone() });
        }

        if let Some(output) = output {
            assert_eq!(output.len(), self.dims[self.dims.len() - 1]);
            output.copy_from_slice(&last);
        }
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

    fn node_derivative(&self, node_id: NodeID, respect_weight: Option<WeightID>, respect_bias: Option<NodeID>, network_state: &NetworkState) -> f64 {
        // Must only be with respect to weight OR bias, not both or neither
        assert!(!(respect_weight.is_some() && respect_bias.is_some()));
        assert!(!(!respect_weight.is_some() && !respect_bias.is_some()));

        let no_activation_node = |layer: usize, node: usize| {
            // input layer never has activation function applied
            if layer == 0 {
                return network_state.layers[0].nodes[node];
            }

            // layer - 1 for second arg because first weights are tied to second layer
            assert_eq!(network_state.layers[layer - 1].nodes.len(), self.hidden[layer - 1].weights[node].len());

            let sum: f64 = network_state.layers[layer - 1].nodes.iter()
                .zip(self.hidden[layer - 1].weights[node].iter())
                .map(|(node, weight)| node * weight)
                .sum();

            sum + self.hidden[layer - 1].biases[node]
        };

        fn derivative_sig(x: f64) -> f64 {
            let a = (-x).exp();
            a / ((1.0 + a) * (1.0 + a))
        }

        // Layer and node index stored in NodeID for both weight and bias, weight index in option for weight only
        let (respect, weight_option) = if let Some(respect) = respect_weight {
            (NodeID { layer: respect.layer, node: respect.node }, Some(respect.weight))
        } else {
            (respect_bias.unwrap(), None)
        };

        match node_id.layer as i64 - respect.layer as i64 {
            // weight is 2+ layers upstream from node in question
            2.. => {
                let a = derivative_sig(no_activation_node(node_id.layer, node_id.node));
                // sum of derivatives of previous nodes times their weights
                let b: f64 = self.hidden[node_id.layer - 1].weights[node_id.node].iter().enumerate()
                    .map(|(index, weight)| {
                        self.node_derivative(NodeID { layer: node_id.layer - 1, node: index }, respect_weight, respect_bias, network_state) * weight
                    })
                    .sum();

                a * b
            },
            // weight is 1 layer upstream from node in question
            1 => {
                let a = derivative_sig(no_activation_node(node_id.layer, node_id.node));
                // derivative of node to which the weight in respect is incoming times the connecting weight to the node in question
                let b = self.node_derivative(NodeID { layer: node_id.layer - 1, node: respect.node }, respect_weight, respect_bias, network_state) * self.hidden[node_id.layer - 1].weights[node_id.node][respect.node];

                a * b
            },
            // weight is in same layer as node in question
            0 => {
                if node_id.node == respect.node {
                    let a = derivative_sig(no_activation_node(node_id.layer, node_id.node));

                    let b = if let Some(weight_idx) = weight_option {
                        network_state.layers[node_id.layer - 1].nodes[weight_idx]
                    } else {
                        1.0
                    };

                    a * b
                } else {
                    0.0
                }
            },
            // weight is downstream from node in question
            _ => 0.0
        }
    }

    fn fast_cost(network_state: &NetworkState, ideal: &[f64]) -> f64 {
        let output = &network_state.layers.last().unwrap().nodes;
        assert_eq!(output.len(), ideal.len());

        output.iter().zip(ideal.iter())
            .map(|(output, ideal)| (output - ideal) * (output - ideal))
            .sum::<f64>()
            .sqrt()
    }

    fn fast_cost_derivative(&self, network_state: &NetworkState, ideal: &[f64], respect_weight: Option<WeightID>, respect_bias: Option<NodeID>) -> f64 {
        let output = &network_state.layers.last().unwrap().nodes;
        assert_eq!(output.len(), ideal.len());

        let a = 0.5 / Self::fast_cost(network_state, ideal);

        let b = output.iter().enumerate().zip(ideal.iter())
            .map(|((i, output), ideal)| {
                2.0 * (output - ideal) * self.node_derivative(NodeID { layer: self.dims.len() - 1, node: i }, respect_weight, respect_bias, network_state)
            })
            .sum::<f64>();

        a * b
    }

    pub fn back_propagate(&mut self, input: &[f64], ideal: &[f64]) -> (f64, f64) {
        const LEARNING_RATE: f64 = 1.0;

        let mut network_state = NetworkState { layers: Vec::new() };
        self.compute_internal(input, None, Some(&mut network_state));


        // let mut function_input = self.gen_derivative_input();

        // let cost = self.cost(input, ideal);
        // let cost_closure = cost.to_closure();
        // let prior_cost = cost_closure(&function_input);
        let prior_cost = Self::fast_cost(&network_state, ideal);

        // back propagate
        for layer in 0..self.hidden.iter().len() {
            // weights
            for node in 0..self.hidden[layer].weights.len() {
                for weight in 0..self.hidden[layer].weights[node].len() {
                    // partial derivative of cost function with respect to current weight
                    // let derivative = cost.derivative(*weight_index);

                    // layer + 1 because first layer of weights informs second layer of nodes
                    let derivative = self.fast_cost_derivative(&network_state, ideal, Some(WeightID { layer: layer + 1, node, weight }), None);
                    // let derivative_closure = derivative.to_closure();

                    // let d_y = derivative_closure(&function_input);
                    let d_y = derivative;
                    // let y = cost_closure(&function_input);
                    let y = Self::fast_cost(&network_state, ideal);

                    if y == 0.0 {
                        continue;
                    }

                    // learning rate, proportional to cost
                    let learning_rate = y * LEARNING_RATE;

                    // shift weight appropriately with the help of newtons method
                    let change = learning_rate * (d_y / y);

                    // println!("{}, {}, {}", d_y, y, change);

                    // function_input[*weight_index] -= change;
                    // if cost_closure(&function_input) > y {
                    //     function_input[*weight_index] += change;
                    // } else {
                    //     *weight -= change;
                    // }

                    self.hidden[layer].weights[node][weight] -= change;
                    // self.compute_internal(input, None, Some(&mut network_state));
                    // if Self::fast_cost(&network_state, ideal) > y {
                    //     self.hidden[layer].weights[node][weight] += change;
                    //     self.compute_internal(input, None, Some(&mut network_state));
                    // }
                }
            }

            // biases
            for node in 0..self.hidden[layer].biases.len() {
                // partial derivative of cost function with respect to current bias
                // let derivative = cost.derivative(*bias_index);
                let derivative = self.fast_cost_derivative(&network_state, ideal, None, Some(NodeID { layer: layer + 1, node }));
                // let derivative_closure = derivative.to_closure();

                // let d_y = derivative_closure(&function_input);
                let d_y = derivative;
                // let y = cost_closure(&function_input);
                let y = Self::fast_cost(&network_state, ideal);

                if y == 0.0 {
                    continue;
                }

                // learning rate, proportional to cost
                let learning_rate = y * LEARNING_RATE;

                // shift bias appropriately with the help of newtons method
                let change = learning_rate * (d_y / y);

                // function_input[*bias_index] -= change;
                // if cost_closure(&function_input) > y {
                //     function_input[*bias_index] += change;
                // } else {
                //     *bias -= change;
                // }

                self.hidden[layer].biases[node] -= change;
                // self.compute_internal(input, None, Some(&mut network_state));
                // if Self::fast_cost(&network_state, ideal) > y {
                //     self.hidden[layer].biases[node] += change;
                //     self.compute_internal(input, None, Some(&mut network_state));
                // }
            }
        }

        // let function_input = self.gen_derivative_input();
        self.compute_internal(input, None, Some(&mut network_state));

        (prior_cost, Self::fast_cost(&network_state, ideal))
    }

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
                Symbol::Func(Function::Sqrt(Box::new(
                    Symbol::Func(Function::Mul(Box::new((
                        Symbol::Func(Function::Add(
                            vec![
                                Symbol::Func(Function::Const(-(*node))),
                                self.node_function(input, self.dims.len() - 1, i, true)
                            ]
                        )),
                        Symbol::Func(Function::Add(
                            vec![
                                Symbol::Func(Function::Const(-(*node))),
                                self.node_function(input, self.dims.len() - 1, i, true)
                            ]
                        ))
                    ))))
                )))
            )
        }

        Symbol::Func(Function::Add(out_nodes_diff))
    }
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// text-based network saving
pub fn save_network_text(path: &str, network: &Network) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    file.write_fmt(format_args!("["))?;
    for layer in network.dims.iter().take(network.dims.len() - 1) {
        file.write_fmt(format_args!("{}, ", layer))?;
    }
    file.write_fmt(format_args!("{}]\n\n", network.dims[network.dims.len() - 1]))?;

    for (i, hidden) in network.hidden.iter().enumerate() {
        file.write_fmt(format_args!("Layer {}:\n", i + 2))?;

        for (i, (weights, bias)) in hidden.weights.iter().zip(hidden.biases.iter()).enumerate() {
            file.write_fmt(format_args!("    Node {}:\n", i + 1))?;
            file.write_fmt(format_args!("        Bias:\n            {}\n", bias))?;
            file.write_fmt(format_args!("        Incoming weights:\n"))?;
            for weight in weights {
                file.write_fmt(format_args!("            {}\n", weight))?;
            }
        }
        file.write_fmt(format_args!("\n"))?;
    }

    Ok(())
}