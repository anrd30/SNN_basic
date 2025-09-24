

````markdown
# RNN to SNN Conversion Project

## Overview
This project demonstrates the conversion of a standard Recurrent Neural Network (RNN) to a Spiking Neural Network (SNN) using a vectorized Leaky Integrate-and-Fire (LIF) model.  
It includes a comparison of **compute cost**, **memory usage**, **spike activity**, and **prediction accuracy** between the RNN and the SNN.
CHECK BRANCHES FOR EEG DATA CLASSIFICATION.

---

## Project Structure
- `rnn/` - Contains the RNN model scripts and pre-trained weights.  
- `utils.py` - Generates synthetic sine-wave datasets for training and evaluation.  
- `rnn_baseline.py` - Defines the baseline RNN architecture.  
- `main.py` - Loads the RNN, converts it to SNN, evaluates performance, and visualizes spikes.  
- `.gitignore` - Ignores unnecessary files like model weights, datasets, and logs.  

---

## Key Functions
- **`utils.generate_sine_data`**: Generates sine-wave sequences for model input/output.  
- **`RNNBaseline`**: A simple RNN with a linear decoder for regression tasks.  
- **`run_rnn()`**: Evaluates the RNN on a batch of input sequences.  
- **`run_snn_vectorized()`**: Converts RNN weights to a vectorized LIF SNN and computes outputs.  
- **`measure_compute()`**: Computes metrics like forward-pass time, FLOPs, memory, and average spike rate.  

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/anrd30/SNN_basic.git
cd SNN_basic
````

2. Install dependencies:

```bash
pip install torch matplotlib numpy
```

3. Run the main script:

```bash
python main.py
```

* This will evaluate the RNN and SNN, print compute metrics, and plot spike activity.

---

## Results

* Compare **MSE**, **FLOPs**, **GPU time**, and **memory usage** for RNN vs SNN.
* Visualize neuron spikes over time using matplotlib.

---

## License

MIT License

```

---


Do you want me to do that?
```
