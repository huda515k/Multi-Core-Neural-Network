# Multi-Core Neural Network Simulation Using Processes and Threads

## Project Overview

This project simulates a neural network architecture on a multi-core processor using operating system-level constructs. Each layer of the neural network is implemented as a separate process, while individual neurons within a layer are represented as threads. This design leverages parallel processing capabilities to enhance computational efficiency during forward and backward signal propagation.

## Architecture

- **Input Layer**: 2 neurons (threads) in 1 process
- **Hidden Layers**: User-specified number of layers, each with user-specified number of neurons (threads) in separate processes
- **Output Layer**: Same number of neurons as hidden layers (threads) in 1 process

## Compilation

```bash
make
```

This will create an executable named `neural_network`.

## Usage

1. Ensure `input.txt` exists with the proper format (see Input File Format below)
2. Run the program:
   ```bash
   ./neural_network
   ```
3. Enter the number of hidden layers when prompted
4. Enter the number of neurons per hidden/output layer when prompted
5. The program will execute:
   - Forward Pass 1: Process input through all layers
   - Backward Pass: Propagate f(x1) and f(x2) backward
   - Forward Pass 2: Process f(x1) and f(x2) as new inputs
6. All outputs are written to `output.txt`

## Input File Format

The `input.txt` file should follow this format:

```
Inputs for 2 neurons of input layer
1.2, 0.5
Weights for 2 neurons of input layer
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
Weights for 8 neurons of hidden layer 1
[8 rows of weights, each with 8 comma-separated values]
Weights for 8 neurons of hidden layer 2
[8 rows of weights, each with 8 comma-separated values]
...
Weights for 8 neurons of output layer
[8 rows of weights, each with 8 comma-separated values]
```

## Implementation Details

### Process-Thread Mapping
- Each layer = 1 process (created via `fork()`)
- Each neuron = 1 thread (created via `pthread_create()`)

### Inter-Process Communication
- Pipes are used for communication between processes
- Forward pass: Data flows from input → hidden layers → output
- Backward pass: f(x1) and f(x2) flow from output → hidden layers → input

### Synchronization
- Mutexes protect shared output arrays
- Semaphores coordinate thread completion
- Thread-safe access to shared resources

### Forward Pass
1. Input layer neurons compute weighted sums of inputs
2. Outputs are aggregated and sent via pipe to first hidden layer
3. Each hidden layer:
   - Reads inputs from previous layer via pipe
   - Spawns threads for each neuron
   - Each neuron computes weighted sum
   - Aggregates outputs and sends to next layer
4. Output layer computes weighted sums and calculates:
   - f(x1) = (output² + output + 1) / 2
   - f(x2) = (output² - output) / 2

### Backward Pass
- f(x1) and f(x2) are propagated backward through layers (for demonstration)
- Once received at input layer, they become new inputs for Forward Pass 2

## Output

All outputs are written to `output.txt`, including:
- Forward Pass 1 outputs for each layer
- Backward Pass signals (f(x1), f(x2))
- Forward Pass 2 outputs for each layer

## Cleanup

The program properly:
- Closes all file descriptors
- Closes all pipes
- Destroys semaphores and mutexes
- Cleans up resources

## Requirements

- C++11 or later
- pthread library
- Linux/Ubuntu environment (for demo)
- POSIX-compliant system

## Notes

- The program is not hardcoded for a fixed number of layers or neurons
- Weights and inputs are read from `input.txt`, not hardcoded
- Backward pass is for demonstration only (no actual weight updates)

