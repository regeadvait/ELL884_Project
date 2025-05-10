# Parameter-Efficient Table QA with LoRA Adapters

This repository contains the implementation of a parameter-efficient Table Question Answering system using LoRA adapters on a frozen BART-Large backbone. The model answers free-form queries over semi-structured tables (FeTaQA dataset) while updating only 1.56% of the full model parameters.

## Features

* *LoRA Adapters*: Low-rank adapters injected into each attention layer for efficient fine-tuning.
* *Modular Design*: Separate adapters for tabular and textual inputs.
* *Unified Input*: Question, page title, and flattened table concatenated into a single text sequence.

## Requirements

* Python 3.8+
* PyTorch
* Transformers
* PEFT (Parameter-Efficient Fine-Tuning)
* Datasets, sacrebleu, rouge\_score, bert-score

Install via:

bash
pip install -r requirements.txt


## Usage

1. *Data Preparation*: Load and preprocess FeTaQA via prepare_data_for_model.
2. *Training*:

   bash
   python train_lora_fetaqa.py \
     --dataset DongfuJiang/FeTaQA \
     --adapter_rank 8 --lr 1e-4 --epochs 3
   
3. *Evaluation*:

   bash
   python eval_lora_fetaqa.py --model_dir ./fetaqa_bart_lora
   

## Citation

If you use this code, please cite:


@article{dasgupta2025lora,
  title={Parameter-Efficient Table Question Answering with LoRA Adapters},
  author={Dasgupta, Ritam and Rege, Advait},
  year={2025}
}
