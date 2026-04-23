import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # 1. Add special tokens (fixed order)
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        # 2. Collect unique words
        unique_words = set()
        
        for text in texts:
            tokens = text.lower().split()
            unique_words.update(tokens)
        
        # 3. Sort for deterministic vocab
        sorted_words = sorted(unique_words)
        
        # 4. Assign IDs starting from 4
        start_idx = len(special_tokens)
        
        for i, word in enumerate(sorted_words):
            idx = start_idx + i
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
        
        # 5. Set vocab size
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.lower().split()
        unk_id = self.word_to_id[self.unk_token]
        
        return [
            self.word_to_id.get(token, unk_id)
            for token in tokens
        ]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        
        for idx in ids:
            word = self.id_to_word.get(idx, self.unk_token)
            
            # Skip padding and control tokens
            if word in {
                self.pad_token,
                self.bos_token,
                self.eos_token
            }:
                continue
            
            words.append(word)
        
        return " ".join(words)