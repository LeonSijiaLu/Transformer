import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, ds,
                 seq_len: int,
                 tokenizer_source: Tokenizer,
                 tokenizer_target: Tokenizer,
                 lang_source: str,
                 lang_target: str):
        super().__init__()

        self.ds = ds
        self.seq_len = seq_len

        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target

        self.lang_source = lang_source
        self.lang_target = lang_target

        # we will only use these tensors in torch.concat()
        # torch.concat() will allocate separate memory, so changes to new concatenated results won't update original tensors
        self.sos_token = torch.tensor([self.tokenizer_source.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_source.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_source.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text_source = self.ds[idx]['translation'][self.lang_source]
        text_target = self.ds[idx]['translation'][self.lang_target]

        # 1. use tokenizer to encode text into tokens
        # .encode() returns an object, ["Hello", ",", "world", "!"]
        # .encode().ids return a list of ids, [101, 113, 203, 119]
        tokens_source = self.tokenizer_source.encode(text_source).ids
        tokens_target = self.tokenizer_target.encode(text_target).ids

        # 2. Count number of padding tokens
        # encoder: each token needs [SOS] and [EOS], so -2
        # decoder: each token needs [SOS], so -1
        padding_tokens_num_source = self.seq_len - len(tokens_source) - 2
        padding_tokens_num_target = self.seq_len - len(tokens_source) - 1

        # 3. Create complete tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tokens_source, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * padding_tokens_num_source)
            ], dim=0
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tokens_target, dtype=torch.int64),
                torch.tensor([self.pad_token] * padding_tokens_num_target)
            ], dim=0
        )

        label = torch.cat(
            [
                torch.tensor(tokens_target, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * padding_tokens_num_target)
            ], dim=0
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # mask tracks at what index a token is not a padding
        encoder_input_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)

        # (1, seq_len) & (1, seq_len, seq_len)
        # it ensures all padding are 0s
        decoder_input_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & tril_mask(len(decoder_input)),

        return {
            "encoder_input": encoder_input,
            "encoder_input_mask": encoder_input_mask,
            "decoder_input": decoder_input,
            "decoder_input_mask": decoder_input_mask,
            "label": label,
            "text_source": text_source,
            "text_target": text_target,
        }
def tril_mask(size):
    mask = torch.tril(torch.ones(1, size, size), diagonal=0).type(torch.int)
    return mask != 0
