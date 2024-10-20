import torch
import torch.nn as nn
from tokenizers.trainers import BpeTrainer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from model import build_transformer
from datasets import load_dataset
from dataset import TranslationDataset, tril_mask
from config import get_config
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

config = get_config()
ds = load_dataset('opus_books',  f"{config['lang_src']}-{config['lang_tgt']}", split='train')

def get_by_language(lang):
    for sample in ds:
        yield sample['translation'][lang]

def get_tokenizer(lang):
    tokenizer = Tokenizer(models.BPE()) # byte pair encoding
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # split by whitespaces

    # unknown, starting of sentence, ending of sentence, padding, mask, separator
    trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]", "[MASK]", "[SEP]"],
                                  min_frequency=2)
    tokenizer.train_from_iterator(iterator=get_by_language(lang), trainer=trainer)

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    tokenizer.save(str(tokenizer_path))
    return tokenizer

def get_dataset():
    tokenizer_source = get_tokenizer(config['lang_src'])
    tokenizer_target = get_tokenizer(config['lang_tgt'])

    train_ds_raw, val_ds_raw = random_split(dataset=ds, lengths=[int(0.9 * len(ds)), int(0.1 * len(ds))])

    train_ds = TranslationDataset(ds=train_ds_raw,
                                  seq_len=config['seq_len'],
                                  tokenizer_source=tokenizer_source,
                                  tokenizer_target=tokenizer_target,
                                  lang_source=config['lang_src'],
                                  lang_target=config['lang_tgt'])

    train_dataloader = DataLoader(dataset=train_ds, batch_size=config['batch_size'], shuffle=True)

    val_ds = TranslationDataset(ds=val_ds_raw,
                                  seq_len=config['seq_len'],
                                  tokenizer_source=tokenizer_source,
                                  tokenizer_target=tokenizer_target,
                                  lang_source=config['lang_src'],
                                  lang_target=config['lang_tgt'])

    val_dataloader = DataLoader(dataset=val_ds, batch_size=1, shuffle=True)

    return tokenizer_source, tokenizer_target, train_dataloader, val_dataloader

def train():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    tokenizer_source, tokenizer_target, train_dataloader, val_dataloader = get_dataset()

    transformer = build_transformer(
        src_vocab_size=tokenizer_source.get_vocab_size(),
        src_seq_len=config["seq_len"],
        target_vocab_size=tokenizer_target.get_vocab_size(),
        target_seq_len=config["seq_len"],
        d_embed=config["d_embed"]
    )

    writer = SummaryWriter(config['experiment_name'])

    initial_epoch = 0
    global_step = 0

    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range (initial_epoch, config['num_epochs']):
