import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
from collections import Counter
import spacy
import random
import time
import math
import numpy as np
from tqdm import tqdm
import pickle
import os
import argparse
from typing import Dict, List, Tuple

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class TranslationDataset(Dataset):
    """Custom dataset for machine translation"""
    def __init__(self, data, src_tokenizer, tgt_tokenizer, src_vocab, tgt_vocab, max_len=50):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['translation']['en']
        tgt_text = item['translation']['fr']
        
        # Tokenize texts
        src_tokens = [tok.text.lower() for tok in self.src_tokenizer(src_text)]
        tgt_tokens = [tok.text.lower() for tok in self.tgt_tokenizer(tgt_text)]
        
        # Reverse source tokens (helps with attention)
        src_tokens = src_tokens[::-1]
        
        # Truncate if necessary
        src_tokens = src_tokens[:self.max_len-2]
        tgt_tokens = tgt_tokens[:self.max_len-2]
        
        # Add special tokens
        src_indices = [self.src_vocab['<sos>']] + [self.src_vocab.get(token, self.src_vocab['<unk>']) 
                                                  for token in src_tokens] + [self.src_vocab['<eos>']]
        tgt_indices = [self.tgt_vocab['<sos>']] + [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) 
                                                  for token in tgt_tokens] + [self.tgt_vocab['<eos>']]
        
        # Pad sequences
        src_indices = src_indices + [self.src_vocab['<pad>']] * (self.max_len - len(src_indices))
        tgt_indices = tgt_indices + [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_indices))
        
        return {
            'src': torch.tensor(src_indices),
            'tgt': torch.tensor(tgt_indices)
        }

def build_vocabulary(texts: List[str], tokenizer, max_size: int) -> Dict[str, int]:
    """Build vocabulary from texts with a maximum size"""
    counter = Counter()
    for text in tqdm(texts, desc="Building vocabulary"):
        tokens = [tok.text.lower() for tok in tokenizer(text)]
        counter.update(tokens)
    
    # Get most common words
    most_common = counter.most_common(max_size - 4)  # -4 for special tokens
    
    # Create vocabulary with special tokens
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    vocab.update({word: idx + 4 for idx, (word, _) in enumerate(most_common)})
    
    return vocab

class Encoder(nn.Module):
    """Encoder with bidirectional GRU"""
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        
        # outputs = [batch_size, src_len, enc_hid_dim * 2]
        # hidden = [2, batch_size, enc_hid_dim]
        outputs, hidden = self.rnn(embedded)
        
        # hidden = [batch_size, dec_hid_dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        return outputs, hidden

class Attention(nn.Module):
    """Attention mechanism"""
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # energy = [batch_size, src_len, attn_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        # mask attention scores
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    """Decoder with attention"""
    def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float, attention: Attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input = [batch_size, 1]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]
        
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # attention = [batch_size, src_len]
        attention = self.attention(hidden, encoder_outputs, mask)
        
        # attention = [batch_size, 1, src_len]
        attention = attention.unsqueeze(1)
        
        # context = [batch_size, 1, enc_hid_dim * 2]
        context = torch.bmm(attention, encoder_outputs)
        
        # rnn_input = [batch_size, 1, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # output = [batch_size, 1, dec_hid_dim]
        # hidden = [1, batch_size, dec_hid_dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        # embedded = [batch_size, emb_dim]
        embedded = embedded.squeeze(1)
        # output = [batch_size, dec_hid_dim]
        output = output.squeeze(1)
        # context = [batch_size, enc_hid_dim * 2]
        context = context.squeeze(1)
        
        # prediction = [batch_size, output_dim]
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    """Sequence-to-sequence model with attention"""
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        mask = (src != 0).float()
        return mask
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        # src = [batch_size, src_len]
        # tgt = [batch_size, tgt_len]
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]
        # hidden = [batch_size, dec_hid_dim]
        encoder_outputs, hidden = self.encoder(src)
        
        # first input to the decoder is the <sos> token
        input = tgt[:, 0]
        
        # create mask for attention
        mask = self.create_mask(src)
        
        for t in range(1, tgt_len):
            # predict and get next input
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1
            
        return outputs

def train(model: nn.Module, iterator: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, clip: float, device: torch.device) -> float:
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator, desc='Training'):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Evaluate the model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc='Evaluating'):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            output = model(src, tgt, 0)  # turn off teacher forcing
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--src_vocab_size', type=int, default=10000)
    parser.add_argument('--tgt_vocab_size', type=int, default=10000)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--enc_hid_dim', type=int, default=512)
    parser.add_argument('--dec_hid_dim', type=int, default=512)
    parser.add_argument('--attn_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of training samples to use')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load spacy models
    try:
        spacy_en = spacy.load('en_core_web_sm')
        spacy_fr = spacy.load('fr_core_news_sm')
    except OSError:
        print("Please install spacy models: python -m spacy download en_core_web_sm fr_core_news_sm")
        return
    
    # Load dataset
    print("Loading WMT14 dataset...")
    dataset = load_dataset("wmt14", "fr-en")
    
    # Limit dataset size if max_samples is specified
    if args.max_samples:
        print(f"Limiting dataset to {args.max_samples} training samples...")
        dataset['train'] = dataset['train'].select(range(min(args.max_samples, len(dataset['train']))))
        val_samples = args.max_samples // 10  # Use 10% of training size for validation
        dataset['validation'] = dataset['validation'].select(range(min(val_samples, len(dataset['validation']))))
        print(f"Training samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
    
    # Build or load vocabularies
    vocab_dir = 'vocabs'
    os.makedirs(vocab_dir, exist_ok=True)
    
    vocab_path_en = os.path.join(vocab_dir, 'vocab_en.pkl')
    vocab_path_fr = os.path.join(vocab_dir, 'vocab_fr.pkl')
    
    if os.path.exists(vocab_path_en) and os.path.exists(vocab_path_fr):
        print("Loading existing vocabularies...")
        with open(vocab_path_en, 'rb') as f:
            en_vocab = pickle.load(f)
        with open(vocab_path_fr, 'rb') as f:
            fr_vocab = pickle.load(f)
    else:
        print("Building vocabularies...")
        en_vocab = build_vocabulary(
            [item['translation']['en'] for item in dataset['train']], 
            spacy_en, 
            args.src_vocab_size
        )
        fr_vocab = build_vocabulary(
            [item['translation']['fr'] for item in dataset['train']], 
            spacy_fr, 
            args.tgt_vocab_size
        )
        
        # Save vocabularies
        with open(vocab_path_en, 'wb') as f:
            pickle.dump(en_vocab, f)
        with open(vocab_path_fr, 'wb') as f:
            pickle.dump(fr_vocab, f)
    
    print(f"English vocabulary size: {len(en_vocab)}")
    print(f"French vocabulary size: {len(fr_vocab)}")
    
    # Create datasets
    train_dataset = TranslationDataset(
        dataset['train'], 
        spacy_en, 
        spacy_fr, 
        en_vocab, 
        fr_vocab,
        args.max_len
    )
    
    valid_dataset = TranslationDataset(
        dataset['validation'], 
        spacy_en, 
        spacy_fr, 
        en_vocab, 
        fr_vocab,
        args.max_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Initialize model
    encoder = Encoder(
        input_dim=len(en_vocab),
        emb_dim=args.emb_dim,
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        dropout=args.dropout
    )
    
    attention = Attention(
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        attn_dim=args.attn_dim
    )
    
    decoder = Decoder(
        output_dim=len(fr_vocab),
        emb_dim=args.emb_dim,
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        dropout=args.dropout,
        attention=attention
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Initialize weights
    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    model.apply(init_weights)
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    
    # Training loop
    best_valid_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, args.clip, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_path = os.path.join(args.save_dir, 'model_best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'en_vocab': en_vocab,
                'fr_vocab': fr_vocab,
                'args': vars(args)
            }, model_path)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    # Save final model
    model_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_loss,
        'en_vocab': en_vocab,
        'fr_vocab': fr_vocab,
        'args': vars(args)
    }, model_path)
    
    print("Training completed!")
    print(f"Best validation loss: {best_valid_loss:.3f}")
    print(f"Models saved in {args.save_dir}")

if __name__ == '__main__':
    main()