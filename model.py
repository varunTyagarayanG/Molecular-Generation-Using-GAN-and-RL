import re
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from layers import Generator, RecurrentDiscriminator
from tokenizer import Tokenizer

RDLogger.DisableLog('rdApp.*')


class MolGen(nn.Module):

    def __init__(self, data, hidden_dim=128, lr=1e-3, device='cpu'):
        """
        MolGen model combining a Generator and a Discriminator.

        Args:
            data (list[str]): List of SMILES strings.
            hidden_dim (int): Hidden dimension size. Defaults to 128.
            lr (float): Learning rate. Defaults to 1e-3.
            device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.
        """
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        self.tokenizer = Tokenizer(data)

        # When constructing the generator we subtract one from the start and end tokens.
        self.generator = Generator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size - 1,
            start_token=self.tokenizer.start_token - 1,  # adjusted for generator indexing
            end_token=self.tokenizer.end_token - 1,
        ).to(device)

        self.discriminator = RecurrentDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,
            bidirectional=True
        ).to(device)

        self.generator_optim = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        self.b = 0.0  # baseline reward for REINFORCE

    def sample_latent(self, batch_size):
        """Sample latent vectors from a standard normal distribution."""
        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def discriminator_loss(self, x, y):
        """Compute binary cross-entropy loss for the discriminator."""
        output = self.discriminator(x)
        y_pred, mask = output['out'], output['mask']
        loss = F.binary_cross_entropy(y_pred, y, reduction='none') * mask
        loss = loss.sum() / mask.sum()
        return loss

    def train_step(self, x):
        """
        A GAN training step: trains the discriminator and then the generator
        using the discriminator’s reward.
        """
        batch_size, len_real = x.size()

        # --- Discriminator Training ---
        x_real = x.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)
        z = self.sample_latent(batch_size)
        generator_outputs = self.generator.forward(z, max_len=20)
        x_gen, log_probs, entropies = generator_outputs.values()
        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        self.discriminator_optim.zero_grad()
        fake_loss = self.discriminator_loss(x_gen, y_gen)
        real_loss = self.discriminator_loss(x_real, y_real)
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()
        clip_grad_value_(self.discriminator.parameters(), 0.1)
        self.discriminator_optim.step()

        # --- Generator Training via GAN Reward ---
        self.generator_optim.zero_grad()
        y_pred, y_pred_mask = self.discriminator(x_gen).values()
        R = (2 * y_pred - 1)
        lengths = y_pred_mask.sum(1).long()
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):
            reward_baseline = reward - self.b
            generator_loss.append((- reward_baseline * log_p).sum())
        generator_loss = torch.stack(generator_loss).mean() - sum(entropies) * 0.01 / batch_size

        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward

        generator_loss.backward()
        clip_grad_value_(self.generator.parameters(), 0.1)
        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward.item()}

    def reinforce_train_step(self, batch_size, reward_function):
        """
        An explicit reinforcement learning update for the generator using REINFORCE.
        Uses an external reward function (e.g., based on QED).
        """
        z = self.sample_latent(batch_size)
        output = self.generator.forward(z, max_len=20)
        x_gen = output['x']  # [B, max_len] (padded)
        log_probs_list = output['log_probabilities']  # list of tensors (per sample)
        entropies_list = output['entropies']          # list of average entropies per sample

        # Convert generated token sequences to SMILES strings.
        generated_smiles = []
        for i in range(x_gen.shape[0]):
            seq = x_gen[i].cpu().numpy().tolist()
            valid_length = sum(1 for token in seq if token != 0)
            seq_trimmed = seq[:valid_length]
            s = self.get_mapped(seq_trimmed)
            generated_smiles.append(s)

        rewards = [reward_function(s) for s in generated_smiles]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        losses = []
        for i, log_p in enumerate(log_probs_list):
            sample_log_prob = log_p.sum()  # total log-probability of sample i
            loss_i = - (rewards_tensor[i] - self.b) * sample_log_prob
            losses.append(loss_i)
        losses_tensor = torch.stack(losses).mean()
        entropy_bonus = sum(entropies_list) * 0.01 / batch_size
        total_loss = losses_tensor - entropy_bonus

        self.generator_optim.zero_grad()
        total_loss.backward()
        clip_grad_value_(self.generator.parameters(), 0.1)
        self.generator_optim.step()

        with torch.no_grad():
            mean_reward = rewards_tensor.mean()
            self.b = 0.9 * self.b + 0.1 * mean_reward

        return {'reinforce_loss': total_loss.item(), 'mean_reward': mean_reward.item()}

    def train_n_steps(self, train_loader, max_step=10000, evaluate_every=50):
        """
        Standard adversarial (GAN) training loop.
        """
        iter_loader = iter(train_loader)
        for step in range(max_step):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
            self.train_step(batch)
            if step % evaluate_every == 0:
                self.eval()
                score = self.evaluate_n(100)
                self.train()
                print(f'GAN step {step:4d}: Validity = {score:.4f}')

    def train_rl_n_steps(self, reward_function, num_steps=1000, batch_size=128, evaluate_every=50):
        """
        Explicit RL training loop using REINFORCE.
        """
        for step in range(num_steps):
            metrics = self.reinforce_train_step(batch_size, reward_function)
            if step % evaluate_every == 0:
                self.eval()
                eval_metrics = self.evaluate_rl(100, reward_function)
                self.train()
                print(f"RL step {step:4d}: RL Loss = {metrics['reinforce_loss']:.4f}, "
                      f"Train Mean Reward = {metrics['mean_reward']:.4f}, "
                      f"Eval Mean Reward = {eval_metrics['mean_reward']:.4f}, "
                      f"Validity = {eval_metrics['validity']:.4f}")

    def create_dataloader(self, data, batch_size=128, shuffle=True, num_workers=5):
        """
        Create a DataLoader for the list of SMILES strings.
        """
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenizer.batch_tokenize,
            num_workers=num_workers
        )

    def get_mapped(self, seq):
        """
        Convert a sequence of token ids to a SMILES string.
        This version removes the special tokens (<pad>, <sos>, and <eos>) 
        by converting the token ids to their corresponding strings and then 
        applying a regular expression to remove any residual occurrences.
        """
        # Convert token ids to tokens using the tokenizer's inverse mapping.
        tokens = [self.tokenizer.inv_mapping.get(i, "") for i in seq]
        # First filter out tokens that are exactly the special tokens.
        tokens = [t for t in tokens if t not in ['<pad>', '<sos>', '<eos>']]
        # Join tokens into a single string.
        joined = "".join(tokens)
        # As an extra safety, remove any occurrences of the special token substrings.
        cleaned = re.sub(r'<(?:pad|sos|eos)>', '', joined)
        return cleaned

    @torch.no_grad()
    def generate_n(self, n):
        """
        Generate n molecules using the generator.
        """
        z = torch.randn((n, self.hidden_dim)).to(self.device)
        x = self.generator(z)['x'].cpu()
        lengths = (x > 0).sum(1)
        # For each sample, consider tokens up to (but not including) the last token.
        return [self.get_mapped(x[:l-1].numpy()) for x, l in zip(x, lengths)]

    def evaluate_n(self, n):
        """
        Evaluate the generator by computing the fraction of valid molecules.
        """
        pack = self.generate_n(n)
        print("Generated examples:", pack[:2])
        valid = np.array([Chem.MolFromSmiles(k) is not None for k in pack])
        return valid.mean()

    def evaluate_rl(self, n, reward_function):
        """
        Evaluate generated molecules using an external reward function.
        """
        pack = self.generate_n(n)
        rewards = []
        valids = []
        for smiles in pack:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    r = reward_function(smiles)
                    valid = 1.0
                else:
                    r = 0.0
                    valid = 0.0
            except Exception:
                r = 0.0
                valid = 0.0
            rewards.append(r)
            valids.append(valid)
        return {'mean_reward': np.mean(rewards), 'validity': np.mean(valids)}


