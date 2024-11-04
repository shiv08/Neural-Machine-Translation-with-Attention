# Neural Machine Translation with Attention

A PyTorch implementation of a sequence-to-sequence neural machine translation system with attention mechanism for English to French translation. The model is trained on the WMT14 dataset and features a bidirectional GRU architecture with Bahdanau attention.

## Model Architecture

### Encoder
- Bidirectional GRU
- Embedding layer with dropout
- Linear transformation to bridge encoder-decoder hidden dimensions
- Batch-first implementation

### Attention Mechanism
- Bahdanau (additive) attention
- Attention dimension: 64 by default
- Masked attention scores for padding
- Energy calculation using tanh activation

### Decoder
- Unidirectional GRU
- Input feeding with context vector
- Combined output projection layer
- Teacher forcing during training

## Implementation Features
- Dataset: WMT14 English-French
- Vocabulary handling with special tokens (`<pad>`, `<unk>`, `<sos>`, `<eos>`)
- Source sentence reversal for better attention learning
- Dynamic batch creation with padding
- Gradient clipping for stable training
- Type hints for better code readability
- Multi-worker data loading

## Training Details

### Hyperparameters
```python
batch_size = 32
embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
attention_dim = 64
dropout = 0.3
learning_rate = 0.001
gradient_clip = 1.0
max_sequence_length = 50
```

### Loss Function & Optimization
- CrossEntropyLoss (ignoring padding tokens)
- Adam optimizer
- Teacher forcing ratio: 0.5

### Training Features
- Best model checkpoint saving
- Training and validation loss monitoring
- Perplexity calculation
- Progress bars using tqdm
- Vocabulary caching for faster retraining

## Results
- Training Loss: ~2.540
- Validation Loss: ~4.409
- Training Perplexity: ~12.679
- Validation Perplexity: ~82.179

## Usage

### Requirements
```bash
torch
spacy
datasets
tqdm
numpy
```

### Spacy Models
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### Training Command
```bash
python train.py --max_samples 100000 \
                --batch_size 32 \
                --epochs 25 \
                --src_vocab_size 32000 \
                --tgt_vocab_size 32000 \
                --emb_dim 256 \
                --enc_hid_dim 512 \
                --dec_hid_dim 512 \
                --attn_dim 64 \
                --dropout 0.3 \
                --clip 0.5 \
                --learning_rate 0.001 \
                --save_dir "checkpoints"
```

### Current Challenges
1. High validation perplexity indicating overfitting
2. Large number of unknown tokens in translations
3. Gap between training and validation metrics

## Project Structure
```
project/
│
├── train.py           # Main training script with model implementation
├── vocabs/           # Directory for cached vocabularies
└── checkpoints/      # Directory for model checkpoints
```

## Future Improvements
1. Implementation of better decoding strategies (currently using basic argmax)
2. Learning rate scheduling
3. Better tokenization and preprocessing
4. Vocabulary size tuning
5. Regularization techniques to combat overfitting

## Hardware Used
- NVIDIA RTX 3060 6GB GPU
- CUDA acceleration

This implementation serves as a foundation for neural machine translation and demonstrates the core concepts of sequence-to-sequence learning with attention. While there's room for improvement in performance metrics, the code provides a clean, typed, and well-structured base for further development.
