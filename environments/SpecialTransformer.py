import torch
import torch.nn as nn
import math
from environments import environment
from consts import *


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module injects some information about the relative or absolute position of the tokens
    in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEncoding module.
        d_model (int): The dimensionality of the model (the input size of the transformer).
        max_len (int): The maximum length of input sequences.
        """
        super().__init__()

        # As shown in the paper, the positional encodings are calculated using sine and cosine functions
        pe = torch.zeros(max_len, d_model).double()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor
        x = x + self.pe[:x.size(0), :]
        return x


class ResidualConnection(nn.Module):
    """
    Residual Connection module adds the input tensor to the output tensor of the module.
    """
    def __init__(self, module: nn.Module):
        super(ResidualConnection, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return x + self.module(x, src_mask=mask)


class specialtransformer_env_ARC(nn.Module):
    """
    This model includes optional fully connected layers, positional encoding and residual connections.
    """
    def __init__(self, config):
        """
        - use_fc (bool): Whether to use the fully connected layer before the transformer.
        - use_positional_encoding (bool): Whether to use positional encoding.
        - use_residuals (bool): Whether to use residual connections.
        - use_norm_first (bool): Whether to use normalization.
        """
        super().__init__()  # Call the __init__ method of the parent class (nn.Module)

        # Read configuration parameters
        input_dim = config["input_dim"]
        dropout = config["dropout"]
        nhead = config["transformer_nheads"]
        hidden_dim = config["hidden_dim"]
        num_layers = config["layers"]
        self.use_fc = config["use_fc"]
        self.use_positional_encoding = config["use_positional_encoding"]
        self.use_residuals = config["use_residuals"]
        self.use_norm_first = config["use_norm_first"]

        # Optional fully connected layer before the transformer encoder
        if self.use_fc:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU()
            ).double()
        else:
            hidden_dim = input_dim

        # Optional positional encoding
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim).double()

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,
                                                                    norm_first=self.use_norm_first).double()

        # Optional residual connections
        if self.use_residuals:
            self.transformer_encoder = nn.ModuleList([
                ResidualConnection(module=self.transformer_encoder_layer)
                for _ in range(num_layers)
            ])
        else:
            self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers).double()

        if config["loss"] == 'CE':
            self.main_task_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)
            ).double()
        else:
            self.main_task_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2),
                nn.LogSoftmax(dim=-1)
            ).double()

    def forward(self, vectors):
        x = vectors["x"]

        # Apply the optional fully connected layer
        if self.use_fc:
            x = self.fc(x)

        # Apply the optional positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)

        output = []
        for i in range(DATA_ROUNDS_PER_GAME):
            if self.use_residuals:
                for layer in self.transformer_encoder:
                    time_output = layer(x[:, :i+1].contiguous())[:, -1, :]
            else:
                time_output = self.transformer_encoder(x[:, :i+1].contiguous())[:, -1, :]
            output.append(time_output)
        output = torch.stack(output, 1)
        output = self.main_task_classifier(output)
        return {"output": output}

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        """
        Predicts probabilities for the given input data.

        Args:
            data (dict): Input data.
            update_vectors (bool): Flag indicating whether to update vectors.
            vectors_in_input (bool): Flag indicating whether vectors are in the input.

        Returns:
            dict: Dictionary containing the model output.
                - "proba" (torch.Tensor): Predicted probabilities.
        """
        assert not update_vectors
        output = self(data)
        output["proba"] = torch.exp(output["output"].flatten())
        return output

    def generate_square_subsequent_mask(self, size):
        """
        Generates a square subsequent mask for masking out future tokens.

        Args:
            size (int): Size of the mask.

        Returns:
            torch.Tensor: Generated mask.
        """
        # Create an upper triangular matrix with -inf on and above the diagonal and 0 below the diagonal
        mask = torch.triu(torch.ones(size, size), 1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0)).double()
        return mask


class specialtransformer_env(environment.Environment):
    def init_model_arc(self, config):
        self.model = specialtransformer_env_ARC(config=config).double()

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            raise NotImplementedError
        output["proba"] = torch.exp(output["output"].flatten())

        return output

    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": 888, }
