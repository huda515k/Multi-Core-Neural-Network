#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>
#include <semaphore.h>
#include <fcntl.h>
#include <cstring>
#include <cmath>

using namespace std;

// Global structures for neural network data
struct LayerData {
    vector<vector<double>> weights;
    vector<double> inputs;
    vector<double> outputs;
};

struct NeuronThreadData {
    int neuron_id;
    vector<double>* inputs;
    vector<double>* weights;
    double* output;
    pthread_mutex_t* output_mutex;
    sem_t* output_sem;
};

// Function declarations
void readInputFile(const string& filename, vector<double>& input_values, 
                   vector<LayerData>& layer_weights);
void* inputNeuronThread(void* arg);
void* hiddenNeuronThread(void* arg);
void* outputNeuronThread(void* arg);
void inputLayerProcess(int num_hidden_neurons, vector<double>& inputs, 
                       vector<vector<double>>& weights, int pipe_write_fd);
void hiddenLayerProcess(int layer_id, int num_neurons, int num_inputs, 
                        int pipe_read_fd, int pipe_write_fd, 
                        vector<vector<double>>& weights);
void outputLayerProcess(int num_neurons, int num_inputs, int pipe_read_fd, 
                        int pipe_write_fd, vector<vector<double>>& weights);
void backwardHiddenLayerProcess(int layer_id, int backward_read_fd, int backward_write_fd);
void backwardInputLayerProcess(int backward_read_fd, int parent_write_fd);
void writeOutputFile(const string& filename, const vector<double>& outputs, 
                     const string& phase);

int main() {
    int num_hidden_layers, num_neurons_per_layer;
    
    cout << "Enter number of hidden layers: ";
    cin >> num_hidden_layers;
    
    cout << "Enter number of neurons per hidden/output layer: ";
    cin >> num_neurons_per_layer;
    
    // Read input file
    vector<double> input_values;
    vector<LayerData> layer_weights;
    readInputFile("input.txt", input_values, layer_weights);
    
    if (input_values.size() != 2) {
        cerr << "Error: Input layer must have exactly 2 neurons" << endl;
        return 1;
    }
    
    // Create pipes for communication
    // Pipe from input to first hidden layer
    int input_to_hidden[2];
    pipe(input_to_hidden);
    
    // Pipes between hidden layers
    vector<int*> hidden_pipes;
    for (int i = 0; i < num_hidden_layers - 1; i++) {
        int* p = new int[2];
        pipe(p);
        hidden_pipes.push_back(p);
    }
    
    // Pipe from last hidden to output
    int hidden_to_output[2];
    pipe(hidden_to_output);
    
    // Pipe from output back to last hidden (for backward pass)
    int output_to_hidden[2];
    pipe(output_to_hidden);
    
    // Pipes for backward pass between hidden layers
    vector<int*> backward_pipes;
    for (int i = 0; i < num_hidden_layers - 1; i++) {
        int* p = new int[2];
        pipe(p);
        backward_pipes.push_back(p);
    }
    
    // Pipe from first hidden back to input (for backward pass)
    int hidden_to_input[2];
    pipe(hidden_to_input);
    
    // Create output file
    ofstream outfile("output.txt", ios::trunc);
    outfile.close();
    
    // FORWARD PASS 1
    cout << "\n========== FORWARD PASS 1 ==========" << endl;
    writeOutputFile("output.txt", input_values, "Forward Pass 1 - Input Layer");
    
    // Fork input layer process
    pid_t input_pid = fork();
    if (input_pid == 0) {
        close(input_to_hidden[0]);
        inputLayerProcess(num_neurons_per_layer, input_values, 
                         layer_weights[0].weights, input_to_hidden[1]);
        close(input_to_hidden[1]);
        exit(0);
    }
    
    // Fork hidden layer processes
    vector<pid_t> hidden_pids;
    for (int i = 0; i < num_hidden_layers; i++) {
        pid_t hidden_pid = fork();
        if (hidden_pid == 0) {
            int read_fd, write_fd;
            
            if (i == 0) {
                // First hidden layer reads from input
                close(input_to_hidden[1]);
                read_fd = input_to_hidden[0];
                if (num_hidden_layers > 1) {
                    close(hidden_pipes[0][0]);
                    write_fd = hidden_pipes[0][1];
                } else {
                    close(hidden_to_output[0]);
                    write_fd = hidden_to_output[1];
                }
            } else if (i == num_hidden_layers - 1) {
                // Last hidden layer writes to output
                close(hidden_pipes[i-1][1]);
                read_fd = hidden_pipes[i-1][0];
                close(hidden_to_output[0]);
                write_fd = hidden_to_output[1];
            } else {
                // Middle hidden layers
                close(hidden_pipes[i-1][1]);
                read_fd = hidden_pipes[i-1][0];
                close(hidden_pipes[i][0]);
                write_fd = hidden_pipes[i][1];
            }
            
            int num_inputs = (i == 0) ? 2 : num_neurons_per_layer;
            hiddenLayerProcess(i, num_neurons_per_layer, num_inputs, 
                              read_fd, write_fd, layer_weights[i+1].weights);
            exit(0);
        }
        hidden_pids.push_back(hidden_pid);
    }
    
    // Fork output layer process
    pid_t output_pid = fork();
    if (output_pid == 0) {
        close(hidden_to_output[1]);
        close(output_to_hidden[0]);
        outputLayerProcess(num_neurons_per_layer, num_neurons_per_layer,
                          hidden_to_output[0], output_to_hidden[1],
                          layer_weights[num_hidden_layers + 1].weights);
        close(hidden_to_output[0]);
        close(output_to_hidden[1]);
        exit(0);
    }
    
    // Wait for all processes
    waitpid(input_pid, NULL, 0);
    for (pid_t pid : hidden_pids) {
        waitpid(pid, NULL, 0);
    }
    waitpid(output_pid, NULL, 0);
    
    // BACKWARD PASS
    cout << "\n========== BACKWARD PASS ==========" << endl;
    
    // Create pipe for input layer to send final values back to parent
    int input_to_parent[2];
    pipe(input_to_parent);
    
    // Close unused pipe ends in parent
    close(output_to_hidden[1]);
    
    // Fork backward pass processes for hidden layers (in reverse order)
    vector<pid_t> backward_hidden_pids;
    if (num_hidden_layers > 0) {
        for (int i = num_hidden_layers - 1; i >= 0; i--) {
            pid_t backward_pid = fork();
            if (backward_pid == 0) {
                int backward_read_fd, backward_write_fd;
                
                if (i == num_hidden_layers - 1) {
                    // Last hidden layer reads from output layer
                    backward_read_fd = output_to_hidden[0];
                    // Close unused ends in this child
                    close(output_to_hidden[1]);
                    if (num_hidden_layers > 1) {
                        // Write to previous hidden layer via backward_pipes
                        close(backward_pipes[i-1][0]);
                        backward_write_fd = backward_pipes[i-1][1];
                        // Close other backward_pipes ends
                        for (int j = 0; j < num_hidden_layers - 1; j++) {
                            if (j != i-1) {
                                close(backward_pipes[j][0]);
                                close(backward_pipes[j][1]);
                            }
                        }
                        close(hidden_to_input[0]);
                        close(hidden_to_input[1]);
                    } else {
                        // Only one hidden layer, write directly to input
                        close(hidden_to_input[0]);
                        backward_write_fd = hidden_to_input[1];
                    }
                } else if (i == 0) {
                    // First hidden layer writes to input layer
                    close(backward_pipes[i][1]);
                    backward_read_fd = backward_pipes[i][0];
                    close(hidden_to_input[0]);
                    backward_write_fd = hidden_to_input[1];
                    // Close unused ends
                    close(output_to_hidden[0]);
                    close(output_to_hidden[1]);
                    for (int j = 0; j < num_hidden_layers - 1; j++) {
                        if (j != i) {
                            close(backward_pipes[j][0]);
                            close(backward_pipes[j][1]);
                        }
                    }
                } else {
                    // Middle hidden layers
                    close(backward_pipes[i][1]);
                    backward_read_fd = backward_pipes[i][0];
                    close(backward_pipes[i-1][0]);
                    backward_write_fd = backward_pipes[i-1][1];
                    // Close unused ends
                    close(output_to_hidden[0]);
                    close(output_to_hidden[1]);
                    close(hidden_to_input[0]);
                    close(hidden_to_input[1]);
                    for (int j = 0; j < num_hidden_layers - 1; j++) {
                        if (j != i && j != i-1) {
                            close(backward_pipes[j][0]);
                            close(backward_pipes[j][1]);
                        }
                    }
                }
                
                backwardHiddenLayerProcess(i, backward_read_fd, backward_write_fd);
                exit(0);
            }
            backward_hidden_pids.push_back(backward_pid);
        }
    }
    
    // Fork backward pass process for input layer
    pid_t backward_input_pid = -1;
    if (num_hidden_layers > 0) {
        backward_input_pid = fork();
        if (backward_input_pid == 0) {
            // Close unused ends in input layer child
            close(hidden_to_input[1]);
            close(input_to_parent[0]);
            close(output_to_hidden[0]);
            close(output_to_hidden[1]);
            for (int j = 0; j < num_hidden_layers - 1; j++) {
                close(backward_pipes[j][0]);
                close(backward_pipes[j][1]);
            }
            backwardInputLayerProcess(hidden_to_input[0], input_to_parent[1]);
            close(hidden_to_input[0]);
            close(input_to_parent[1]);
            exit(0);
        }
        close(input_to_parent[1]);
        // Close unused ends in parent
        close(hidden_to_input[0]);
        close(hidden_to_input[1]);
        for (int j = 0; j < num_hidden_layers - 1; j++) {
            close(backward_pipes[j][0]);
            close(backward_pipes[j][1]);
        }
    }
    
    // Wait for all backward pass processes
    for (pid_t pid : backward_hidden_pids) {
        waitpid(pid, NULL, 0);
    }
    
    // Read final backward signal values from input layer
    double fx1, fx2;
    if (num_hidden_layers > 0) {
        read(input_to_parent[0], &fx1, sizeof(double));
        read(input_to_parent[0], &fx2, sizeof(double));
        close(input_to_parent[0]);
        waitpid(backward_input_pid, NULL, 0);
        // Close remaining pipe ends
        close(output_to_hidden[0]);
    } else {
        // No hidden layers - read directly from output
        read(output_to_hidden[0], &fx1, sizeof(double));
        read(output_to_hidden[0], &fx2, sizeof(double));
        close(output_to_hidden[0]);
    }
    
    cout << "Backward signal received at input layer: f(x1)=" << fx1 
         << ", f(x2)=" << fx2 << endl;
    
    // FORWARD PASS 2 (using f(x1) and f(x2) as new inputs)
    cout << "\n========== FORWARD PASS 2 ==========" << endl;
    vector<double> new_inputs = {fx1, fx2};
    writeOutputFile("output.txt", new_inputs, "Forward Pass 2 - Input Layer");
    
    // Create new pipes for second forward pass
    pipe(input_to_hidden);
    for (int i = 0; i < num_hidden_layers - 1; i++) {
        pipe(hidden_pipes[i]);
    }
    pipe(hidden_to_output);
    
    // Fork processes again for second forward pass
    input_pid = fork();
    if (input_pid == 0) {
        close(input_to_hidden[0]);
        inputLayerProcess(num_neurons_per_layer, new_inputs, 
                         layer_weights[0].weights, input_to_hidden[1]);
        close(input_to_hidden[1]);
        exit(0);
    }
    
    hidden_pids.clear();
    for (int i = 0; i < num_hidden_layers; i++) {
        pid_t hidden_pid = fork();
        if (hidden_pid == 0) {
            int read_fd, write_fd;
            
            if (i == 0) {
                close(input_to_hidden[1]);
                read_fd = input_to_hidden[0];
                if (num_hidden_layers > 1) {
                    close(hidden_pipes[0][0]);
                    write_fd = hidden_pipes[0][1];
                } else {
                    close(hidden_to_output[0]);
                    write_fd = hidden_to_output[1];
                }
            } else if (i == num_hidden_layers - 1) {
                close(hidden_pipes[i-1][1]);
                read_fd = hidden_pipes[i-1][0];
                close(hidden_to_output[0]);
                write_fd = hidden_to_output[1];
            } else {
                close(hidden_pipes[i-1][1]);
                read_fd = hidden_pipes[i-1][0];
                close(hidden_pipes[i][0]);
                write_fd = hidden_pipes[i][1];
            }
            
            int num_inputs = (i == 0) ? 2 : num_neurons_per_layer;
            hiddenLayerProcess(i, num_neurons_per_layer, num_inputs, 
                              read_fd, write_fd, layer_weights[i+1].weights);
            exit(0);
        }
        hidden_pids.push_back(hidden_pid);
    }
    
    output_pid = fork();
    if (output_pid == 0) {
        close(hidden_to_output[1]);
        outputLayerProcess(num_neurons_per_layer, num_neurons_per_layer,
                          hidden_to_output[0], -1, 
                          layer_weights[num_hidden_layers + 1].weights);
        close(hidden_to_output[0]);
        exit(0);
    }
    
    // Wait for all processes
    waitpid(input_pid, NULL, 0);
    for (pid_t pid : hidden_pids) {
        waitpid(pid, NULL, 0);
    }
    waitpid(output_pid, NULL, 0);
    
    // Cleanup - close all remaining pipe file descriptors
    // Note: Some pipes may already be closed, but closing them again is safe (returns -1 with EBADF)
    close(input_to_hidden[0]);
    close(input_to_hidden[1]);
    for (int i = 0; i < num_hidden_layers - 1; i++) {
        close(hidden_pipes[i][0]);
        close(hidden_pipes[i][1]);
        delete[] hidden_pipes[i];
    }
    close(hidden_to_output[0]);
    close(hidden_to_output[1]);
    close(output_to_hidden[0]);
    close(output_to_hidden[1]);
    for (int i = 0; i < num_hidden_layers - 1; i++) {
        close(backward_pipes[i][0]);
        close(backward_pipes[i][1]);
        delete[] backward_pipes[i];
    }
    close(hidden_to_input[0]);
    close(hidden_to_input[1]);
    
    cout << "\n========== Execution Complete ==========" << endl;
    cout << "All outputs written to output.txt" << endl;
    
    return 0;
}

// Read input file and parse weights
void readInputFile(const string& filename, vector<double>& input_values, 
                   vector<LayerData>& layer_weights) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open input file " << filename << endl;
        exit(1);
    }
    
    // Read all lines into a vector
    vector<string> lines;
    string line;
    while (getline(file, line)) {
        lines.push_back(line);
    }
    file.close();
    
    // Check if file uses section headers format
    bool has_headers = false;
    for (const string& l : lines) {
        if (l.find("Inputs for") != string::npos || 
            l.find("Weights for") != string::npos) {
            has_headers = true;
            break;
        }
    }
    
    if (has_headers) {
        // Parse format with section headers
        string current_section = "";
        
        for (size_t i = 0; i < lines.size(); i++) {
            line = lines[i];
            // Trim whitespace
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == string::npos) continue; // Skip empty lines
            line = line.substr(first);
            
            // Check for section headers
            if (line.find("Inputs for") != string::npos && 
                line.find("input layer") != string::npos) {
                current_section = "inputs";
                continue;
            } else if (line.find("Weights for") != string::npos && 
                       line.find("input layer") != string::npos) {
                current_section = "input_weights";
                LayerData layer;
                layer_weights.push_back(layer);
                continue;
            } else if (line.find("Weights for") != string::npos && 
                       line.find("hidden layer") != string::npos) {
                current_section = "hidden_weights";
                LayerData layer;
                layer_weights.push_back(layer);
                continue;
            } else if (line.find("Weights for") != string::npos && 
                       line.find("output layer") != string::npos) {
                current_section = "output_weights";
                LayerData layer;
                layer_weights.push_back(layer);
                continue;
            }
            
            // Parse data lines
            stringstream ss(line);
            double value;
            vector<double> values;
            
            while (ss >> value) {
                values.push_back(value);
                if (ss.peek() == ',') ss.ignore();
            }
            
            if (values.empty()) continue;
            
            if (current_section == "inputs") {
                input_values.insert(input_values.end(), values.begin(), values.end());
            } else if (current_section == "input_weights" || 
                       current_section == "hidden_weights" || 
                       current_section == "output_weights") {
                if (!layer_weights.empty()) {
                    layer_weights.back().weights.push_back(values);
                }
            }
        }
    } else {
        // Parse simple format without headers
        // Read all non-empty lines
        vector<string> data_lines;
        for (const string& l : lines) {
            size_t first = l.find_first_not_of(" \t\r\n");
            if (first != string::npos) {
                data_lines.push_back(l);
            }
        }
        
        if (data_lines.empty()) {
            cerr << "Error: Input file is empty" << endl;
            exit(1);
        }
        
        // First line: inputs
        stringstream ss(data_lines[0]);
        double value;
        while (ss >> value) {
            input_values.push_back(value);
            if (ss.peek() == ',') ss.ignore();
        }
        
        // Next 2 lines: input layer weights (2 neurons)
        LayerData input_layer;
        for (int i = 1; i <= 2 && i < (int)data_lines.size(); i++) {
            vector<double> weights_row;
            stringstream ss2(data_lines[i]);
            while (ss2 >> value) {
                weights_row.push_back(value);
                if (ss2.peek() == ',') ss2.ignore();
            }
            if (!weights_row.empty()) {
                input_layer.weights.push_back(weights_row);
            }
        }
        layer_weights.push_back(input_layer);
        
        // Remaining lines: hidden layers and output layer weights
        int line_idx = 3; // Start after inputs (line 0) and input weights (lines 1-2)
        
        // Read all remaining weight rows
        vector<vector<double>> all_weight_rows;
        for (int i = line_idx; i < (int)data_lines.size(); i++) {
            vector<double> weights_row;
            stringstream ss2(data_lines[i]);
            while (ss2 >> value) {
                weights_row.push_back(value);
                if (ss2.peek() == ',') ss2.ignore();
            }
            if (!weights_row.empty()) {
                all_weight_rows.push_back(weights_row);
            }
        }
        
        // Calculate number of hidden layers from the data
        // Total weight rows = num_hidden_layers * neurons_per_layer + neurons_per_layer (output)
        // We need to detect neurons_per_layer from the data
        int neurons_per_layer = 0;
        if (!all_weight_rows.empty()) {
            // Count rows until we see a pattern change or reach output layer
            // For now, assume 8 neurons per layer (can be made more dynamic)
            // Actually, we can count consecutive rows with same number of weights
            neurons_per_layer = all_weight_rows[0].size();
        }
        
        if (neurons_per_layer > 0 && all_weight_rows.size() >= neurons_per_layer) {
            // Calculate: (total_rows - output_layer_rows) / neurons_per_layer = hidden_layers
            int total_rows = all_weight_rows.size();
            int num_hidden_layers_from_file = (total_rows - neurons_per_layer) / neurons_per_layer;
            
            // Parse hidden layers
            for (int layer = 0; layer < num_hidden_layers_from_file; layer++) {
                LayerData hidden_layer;
                int start_idx = layer * neurons_per_layer;
                for (int i = 0; i < neurons_per_layer && (start_idx + i) < (int)all_weight_rows.size(); i++) {
                    hidden_layer.weights.push_back(all_weight_rows[start_idx + i]);
                }
                layer_weights.push_back(hidden_layer);
            }
            
            // Last neurons_per_layer rows: output layer
            LayerData output_layer;
            int output_start_idx = num_hidden_layers_from_file * neurons_per_layer;
            for (int i = 0; i < neurons_per_layer && (output_start_idx + i) < (int)all_weight_rows.size(); i++) {
                output_layer.weights.push_back(all_weight_rows[output_start_idx + i]);
            }
            layer_weights.push_back(output_layer);
        }
    }
    
    // Validation
    if (input_values.size() != 2) {
        cerr << "Error: Input layer must have exactly 2 neurons" << endl;
        exit(1);
    }
    
    if (layer_weights.empty()) {
        cerr << "Error: No weights found in input file" << endl;
        exit(1);
    }
}

// Input neuron thread function
void* inputNeuronThread(void* arg) {
    NeuronThreadData* data = (NeuronThreadData*)arg;
    
    double sum = 0.0;
    for (size_t i = 0; i < data->inputs->size() && i < data->weights->size(); i++) {
        sum += (*data->inputs)[i] * (*data->weights)[i];
    }
    
    // Thread-safe output update
    pthread_mutex_lock(data->output_mutex);
    *(data->output) = sum;
    pthread_mutex_unlock(data->output_mutex);
    
    sem_post(data->output_sem);
    
    cout << "Input neuron " << data->neuron_id << " computed output: " << sum << endl;
    
    pthread_exit(NULL);
}

// Hidden neuron thread function
void* hiddenNeuronThread(void* arg) {
    NeuronThreadData* data = (NeuronThreadData*)arg;
    
    double sum = 0.0;
    for (size_t i = 0; i < data->inputs->size() && i < data->weights->size(); i++) {
        sum += (*data->inputs)[i] * (*data->weights)[i];
    }
    
    // Thread-safe output update
    pthread_mutex_lock(data->output_mutex);
    *(data->output) = sum;
    pthread_mutex_unlock(data->output_mutex);
    
    sem_post(data->output_sem);
    
    pthread_exit(NULL);
}

// Output neuron thread function
void* outputNeuronThread(void* arg) {
    NeuronThreadData* data = (NeuronThreadData*)arg;
    
    double sum = 0.0;
    for (size_t i = 0; i < data->inputs->size() && i < data->weights->size(); i++) {
        sum += (*data->inputs)[i] * (*data->weights)[i];
    }
    
    // Thread-safe output update
    pthread_mutex_lock(data->output_mutex);
    *(data->output) = sum;
    pthread_mutex_unlock(data->output_mutex);
    
    sem_post(data->output_sem);
    
    pthread_exit(NULL);
}

// Input layer process
void inputLayerProcess(int /* num_hidden_neurons */, vector<double>& inputs, 
                       vector<vector<double>>& weights, int pipe_write_fd) {
    cout << "Input Layer Process: Starting with " << inputs.size() << " inputs" << endl;
    
    // Create threads for each neuron (2 neurons in input layer)
    const int num_neurons = 2;
    pthread_t threads[num_neurons];
    NeuronThreadData thread_data[num_neurons];
    double outputs[num_neurons];
    pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
    sem_t output_sem;
    sem_init(&output_sem, 0, 0);
    
    // Create threads
    for (int i = 0; i < num_neurons; i++) {
        thread_data[i].neuron_id = i;
        thread_data[i].inputs = &inputs;
        thread_data[i].weights = &weights[i];
        thread_data[i].output = &outputs[i];
        thread_data[i].output_mutex = &output_mutex;
        thread_data[i].output_sem = &output_sem;
        
        pthread_create(&threads[i], NULL, inputNeuronThread, &thread_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_neurons; i++) {
        sem_wait(&output_sem);
        pthread_join(threads[i], NULL);
    }
    
    // Aggregate outputs
    vector<double> aggregated_outputs;
    for (int i = 0; i < num_neurons; i++) {
        aggregated_outputs.push_back(outputs[i]);
    }
    
    // Write outputs to pipe
    int num_outputs = aggregated_outputs.size();
    write(pipe_write_fd, &num_outputs, sizeof(int));
    for (double val : aggregated_outputs) {
        write(pipe_write_fd, &val, sizeof(double));
    }
    
    cout << "Input Layer Process: Sent " << num_outputs << " outputs to next layer" << endl;
    
    // Cleanup
    sem_destroy(&output_sem);
    pthread_mutex_destroy(&output_mutex);
}

// Hidden layer process
void hiddenLayerProcess(int layer_id, int num_neurons, int /* num_inputs */, 
                        int pipe_read_fd, int pipe_write_fd, 
                        vector<vector<double>>& weights) {
    cout << "Hidden Layer " << (layer_id + 1) << " Process: Starting" << endl;
    
    // Read inputs from pipe
    int num_inputs_received;
    read(pipe_read_fd, &num_inputs_received, sizeof(int));
    
    vector<double> inputs;
    for (int i = 0; i < num_inputs_received; i++) {
        double val;
        read(pipe_read_fd, &val, sizeof(double));
        inputs.push_back(val);
    }
    
    cout << "Hidden Layer " << (layer_id + 1) << ": Received " << inputs.size() 
         << " inputs" << endl;
    
    // Write to output file
    writeOutputFile("output.txt", inputs, 
                   "Forward Pass - Hidden Layer " + to_string(layer_id + 1) + " Input");
    
    // Create threads for each neuron
    vector<pthread_t> threads(num_neurons);
    vector<NeuronThreadData> thread_data(num_neurons);
    vector<double> outputs(num_neurons);
    pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
    sem_t output_sem;
    sem_init(&output_sem, 0, 0);
    
    // Create threads
    for (int i = 0; i < num_neurons; i++) {
        thread_data[i].neuron_id = i;
        thread_data[i].inputs = &inputs;
        if (static_cast<size_t>(i) < weights.size()) {
            thread_data[i].weights = &weights[i];
        } else {
            // If not enough weights, use empty vector
            static vector<double> empty_weights;
            thread_data[i].weights = &empty_weights;
        }
        thread_data[i].output = &outputs[i];
        thread_data[i].output_mutex = &output_mutex;
        thread_data[i].output_sem = &output_sem;
        
        pthread_create(&threads[i], NULL, hiddenNeuronThread, &thread_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_neurons; i++) {
        sem_wait(&output_sem);
        pthread_join(threads[i], NULL);
    }
    
    // Aggregate outputs
    vector<double> aggregated_outputs;
    for (int i = 0; i < num_neurons; i++) {
        aggregated_outputs.push_back(outputs[i]);
    }
    
    // Write to output file
    writeOutputFile("output.txt", aggregated_outputs, 
                   "Forward Pass - Hidden Layer " + to_string(layer_id + 1) + " Output");
    
    cout << "Hidden Layer " << (layer_id + 1) << ": Computed " << aggregated_outputs.size() 
         << " outputs" << endl;
    
    // Write outputs to pipe
    if (pipe_write_fd >= 0) {
        int num_outputs = aggregated_outputs.size();
        write(pipe_write_fd, &num_outputs, sizeof(int));
        for (double val : aggregated_outputs) {
            write(pipe_write_fd, &val, sizeof(double));
        }
        cout << "Hidden Layer " << (layer_id + 1) << ": Sent outputs to next layer" << endl;
    }
    
    // Cleanup
    sem_destroy(&output_sem);
    pthread_mutex_destroy(&output_mutex);
}

// Output layer process
void outputLayerProcess(int num_neurons, int /* num_inputs */, int pipe_read_fd, 
                        int pipe_write_fd, vector<vector<double>>& weights) {
    cout << "Output Layer Process: Starting" << endl;
    
    // Read inputs from pipe
    int num_inputs_received;
    read(pipe_read_fd, &num_inputs_received, sizeof(int));
    
    vector<double> inputs;
    for (int i = 0; i < num_inputs_received; i++) {
        double val;
        read(pipe_read_fd, &val, sizeof(double));
        inputs.push_back(val);
    }
    
    cout << "Output Layer: Received " << inputs.size() << " inputs" << endl;
    
    // Write to output file
    writeOutputFile("output.txt", inputs, "Forward Pass - Output Layer Input");
    
    // Create threads for each neuron
    vector<pthread_t> threads(num_neurons);
    vector<NeuronThreadData> thread_data(num_neurons);
    vector<double> outputs(num_neurons);
    pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;
    sem_t output_sem;
    sem_init(&output_sem, 0, 0);
    
    // Create threads
    for (int i = 0; i < num_neurons; i++) {
        thread_data[i].neuron_id = i;
        thread_data[i].inputs = &inputs;
        if (static_cast<size_t>(i) < weights.size()) {
            thread_data[i].weights = &weights[i];
        } else {
            static vector<double> empty_weights;
            thread_data[i].weights = &empty_weights;
        }
        thread_data[i].output = &outputs[i];
        thread_data[i].output_mutex = &output_mutex;
        thread_data[i].output_sem = &output_sem;
        
        pthread_create(&threads[i], NULL, outputNeuronThread, &thread_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_neurons; i++) {
        sem_wait(&output_sem);
        pthread_join(threads[i], NULL);
    }
    
    // Aggregate outputs
    vector<double> aggregated_outputs;
    double total_sum = 0.0;
    for (int i = 0; i < num_neurons; i++) {
        aggregated_outputs.push_back(outputs[i]);
        total_sum += outputs[i];
    }
    
    // Write to output file
    writeOutputFile("output.txt", aggregated_outputs, "Forward Pass - Output Layer Output");
    
    cout << "Output Layer: Computed " << aggregated_outputs.size() << " outputs" << endl;
    cout << "Output Layer: Sum of all outputs = " << total_sum << endl;
    
    // Compute f(x1) and f(x2)
    double fx1 = (total_sum * total_sum + total_sum + 1) / 2.0;
    double fx2 = (total_sum * total_sum - total_sum) / 2.0;
    
    cout << "Output Layer: f(x1) = " << fx1 << ", f(x2) = " << fx2 << endl;
    
    // Write f(x1) and f(x2) to output file
    vector<double> fx_values = {fx1, fx2};
    writeOutputFile("output.txt", fx_values, "Backward Pass - f(x1) and f(x2)");
    
    // Send backward signal through pipe if pipe_write_fd is valid
    if (pipe_write_fd >= 0) {
        write(pipe_write_fd, &fx1, sizeof(double));
        write(pipe_write_fd, &fx2, sizeof(double));
        cout << "Output Layer: Sent backward signal (f(x1), f(x2))" << endl;
    }
    
    // Cleanup
    sem_destroy(&output_sem);
    pthread_mutex_destroy(&output_mutex);
}

// Backward pass hidden layer process
void backwardHiddenLayerProcess(int layer_id, int backward_read_fd, int backward_write_fd) {
    cout << "Backward Pass - Hidden Layer " << (layer_id + 1) << " Process: Starting" << endl;
    
    // Read backward signal from next layer (or output layer)
    double fx1, fx2;
    read(backward_read_fd, &fx1, sizeof(double));
    read(backward_read_fd, &fx2, sizeof(double));
    
    cout << "Backward Pass - Hidden Layer " << (layer_id + 1) 
         << ": Received backward signal f(x1)=" << fx1 << ", f(x2)=" << fx2 << endl;
    
    // Write to output file for backward pass
    vector<double> backward_signal = {fx1, fx2};
    writeOutputFile("output.txt", backward_signal, 
                   "Backward Pass - Hidden Layer " + to_string(layer_id + 1));
    
    // Propagate backward signal to previous layer
    if (backward_write_fd >= 0) {
        write(backward_write_fd, &fx1, sizeof(double));
        write(backward_write_fd, &fx2, sizeof(double));
        cout << "Backward Pass - Hidden Layer " << (layer_id + 1) 
             << ": Propagated backward signal to previous layer" << endl;
    }
}

// Backward pass input layer process
void backwardInputLayerProcess(int backward_read_fd, int parent_write_fd) {
    cout << "Backward Pass - Input Layer Process: Starting" << endl;
    
    // Read backward signal from first hidden layer
    double fx1, fx2;
    read(backward_read_fd, &fx1, sizeof(double));
    read(backward_read_fd, &fx2, sizeof(double));
    
    cout << "Backward Pass - Input Layer: Received backward signal f(x1)=" << fx1 
         << ", f(x2)=" << fx2 << endl;
    
    // Write to output file
    vector<double> backward_signal = {fx1, fx2};
    writeOutputFile("output.txt", backward_signal, "Backward Pass - Input Layer");
    
    // Send values back to parent process for second forward pass
    if (parent_write_fd >= 0) {
        write(parent_write_fd, &fx1, sizeof(double));
        write(parent_write_fd, &fx2, sizeof(double));
        cout << "Backward Pass - Input Layer: Sent values to parent process" << endl;
    }
}

// Write output to file
void writeOutputFile(const string& filename, const vector<double>& outputs, 
                     const string& phase) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) {
        cerr << "Error: Cannot open output file " << filename << endl;
        return;
    }
    
    file << phase << ":" << endl;
    for (size_t i = 0; i < outputs.size(); i++) {
        file << outputs[i];
        if (i < outputs.size() - 1) {
            file << ", ";
        }
    }
    file << endl << endl;
    
    file.close();
}

