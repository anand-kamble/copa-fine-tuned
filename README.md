# Aircraft Fault ATA Prediction using LoRA

This repository contains scripts and configurations to fine-tune a pre-trained language model for predicting the Air Transport Association (ATA) code of a fault in aircraft based on descriptive text. The model utilizes low-rank adaptation (LoRA) to efficiently train on specific tasks with limited computational resources.

## Repository Structure

```
.
├── For Copa
│   ├── config.json         # Configuration file with data and output settings
│   ├── infer.py            # Inference script to predict ATA codes based on input text
│   ├── main.py             # Main script for training and inference
│   ├── outputs             # Directory where model checkpoints and outputs are stored
│   └── train.py            # Script to fine-tune the language model on aircraft fault data
├── infer.py                # Inference script (root-level for quick access)
├── outputs
│   └── checkpoint-100      # Example output directory with model checkpoints
├── README.md               # Documentation for the repository
└── train.py                # Training script (root-level for quick access)
```

## Requirements

Ensure you have the following installed:

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- `bitsandbytes`
- Datasets (for loading and processing training data)

You can install the necessary dependencies using the following command:

```bash
pip install torch transformers bitsandbytes datasets
```

## Purpose

The primary goal of this project is to fine-tune a language model to predict the ATA code of an aircraft fault based on its description. The ATA code is essential for categorizing faults and streamlining maintenance operations in the aviation industry.

## Configuration

The `config.json` file is used to specify the dataset, output format, and other important parameters. Here is an example:

```json
{
  "output_dir": "copa_output",
  "output_format": "parquet",
  "output_file_name": "filtered.parquet",
  "data_files": [
    {
      "fileName": "ISDP LOGBOOK REPORT.csv",
      "fileType": "csv",
      "colOfInterest": ["FLEET", "FAULT_FOUND_DATE", "FAULT_SOURCE", "FAULT_NAME", "FAULT_SDESC", "CORRECTIVE_ACTION", "MAINT_DELAY_TIME_QT", "ATA", "FAULT_SEVERITY"],
      "separator": ","
    }
  ],
  "test_rows": 33000
}
```

This file defines how the data is loaded, processed, and outputted during training and inference.

## Training

The `train.py` script is used to fine-tune a pre-trained model using LoRA. The script is designed to take descriptive text from aircraft maintenance logs and train the model to predict the corresponding ATA code.

### Running the Training Script

```bash
python train.py
```

### Training Script Details

- **Frozen Parameters**: The base model parameters are frozen, and only the LoRA adapters are trained.
- **Mixed Precision Training**: The model uses 8-bit precision and FP16 to reduce memory usage and speed up training.
- **Gradient Checkpointing**: Enabled to reduce memory footprint during backpropagation.
- **Custom Data Processing**: The training data is processed to merge relevant columns into a single string input that the model can learn from, focusing on predicting the ATA code.

## Inference

Once the model is trained, you can generate predictions using the `infer.py` script:

```bash
python infer.py
```

### Inference Script Details

The `infer.py` script loads the fine-tuned model and generates the ATA code based on a given fault description. Key features include:

- **8-bit Precision**: The model is loaded with reduced precision for efficient inference.
- **Prompt-Based Prediction**: Input a fault description, and the model will predict the ATA code.

## Output

The `outputs` directory stores all the checkpoints and generated outputs from both training and inference processes. You can customize the output location by modifying the training and inference scripts.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Fine-tune the model:

   ```bash
   python train.py
   ```

4. Predict ATA codes:

   ```bash
   python infer.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [bitsandbytes](https://github.com/facebookresearch/bitsandbytes)

