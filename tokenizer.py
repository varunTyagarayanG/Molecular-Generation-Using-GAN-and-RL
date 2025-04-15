import torch

class Tokenizer(object):
    def __init__(self, data):
        # Combine all SMILES strings in the provided data into one large string,
        # then get a list of unique characters. We add special tokens: '<eos>' for 
        # end-of-sequence and '<sos>' for start-of-sequence.
        unique_char = list(set(''.join(data))) + ['<eos>'] + ['<sos>']
        
        # Create a mapping dictionary starting with a special token for padding.
        # The '<pad>' token is always assigned to index 0.
        self.mapping = {'<pad>': 0}
        
        # Enumerate through the list of unique characters (including our special tokens),
        # starting the index from 1 because index 0 is already used for '<pad>'.
        # Each character is assigned a unique integer.
        for i, c in enumerate(unique_char, start=1):
            self.mapping[c] = i
        
        # Build an inverse mapping dictionary from indices back to characters.
        self.inv_mapping = {v: k for k, v in self.mapping.items()}
        
        # Set up the start-of-sequence token using the mapping.
        self.start_token = self.mapping['<sos>']
        
        # Set up the end-of-sequence token using the mapping.
        self.end_token = self.mapping['<eos>']
        
        # Save the total vocabulary size which equals the number of unique tokens
        # in the mapping. This includes the padding and both special tokens.
        self.vocab_size = len(self.mapping.keys())
    
    def encode_smile(self, mol, add_eos=True):
        """
        Encodes a SMILES string into a sequence of integer tokens.
        
        Args:
            mol (str): A single SMILES molecule string.
            add_eos (bool): Whether to add the end-of-sequence token to the end of the encoded list.

        Returns:
            torch.LongTensor: Tensor containing the sequence of tokens.
        """
        # For each character in the SMILES string, retrieve its corresponding token from the mapping.
        out = [self.mapping[i] for i in mol]
        
        # If requested, append the end-of-sequence token at the end of the token sequence.
        if add_eos:
            out = out + [self.end_token]
        
        # Convert the list of tokens into a PyTorch LongTensor.
        return torch.LongTensor(out)
    
    def batch_tokenize(self, batch):
        """
        Encodes and pads a batch of SMILES strings into sequences of tokens.
        
        Args:
            batch (list of str): A batch of SMILES molecule strings.
        
        Returns:
            torch.Tensor: A batch-first padded tensor of tokenized sequences.
        """
        # Apply the encode_smile function to each string in the batch using map.
        out = map(lambda x: self.encode_smile(x), batch)
        
        # Convert the iterator to a list and then pad the sequences so that all sequences
        # in the batch have the same length. Padding is done using the '<pad>' token (index 0).
        return torch.nn.utils.rnn.pad_sequence(list(out), batch_first=True)
