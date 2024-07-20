import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from transformers import AutoTokenizer # huggingface tokenizers
from datasets import load_dataset # huggingface datasets


def is_arabic(char):
    return (
        '\u0600' <= char <= '\u06FF' or
        '\u0750' <= char <= '\u077F' or
        '\u08A0' <= char <= '\u08FF' or
        '\uFB50' <= char <= '\uFDFF' or
        '\uFE70' <= char <= '\uFEFF'
    )

def contains_arabic(text):
    return any(is_arabic(char) for char in text)


if __name__ == '__main__':
    
    dataset = load_dataset("ayoubkirouane/Algerian-Darija", split='v1')
    # Filtering out rows where there are no arabic words
    dataset = dataset.filter(lambda s: contains_arabic(s['Text']))
    
    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset.train_test_split(test_size=0.1, seed=2357, shuffle=True) #TODO check seed here
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    
    # Load the base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Train the new tokenizer
    tokenizer = base_tokenizer.train_new_from_iterator(dataset['Text'], vocab_size=10000)
    # Save the new tokenizer
    tokenizer.save_pretrained("dardja_tokenizer")

    def process(sample):
        ids = tokenizer.encode(sample['Text'])
        ids.append(tokenizer.eos_token_id)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['Text'],
        desc="tokenizing the splits",
        num_proc=2
    )
    
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 20 #This value mgiht change later when i get to model training specifics

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
