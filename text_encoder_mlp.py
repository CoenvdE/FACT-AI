import numpy as np
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    A text encoder module that uses a transformer model from a CLIP architecture
    to encode text prompts into feature embeddings.

    Attributes:
        transformer (nn.Module): The transformer module from the CLIP model.
        positional_embedding (Tensor): The positional embeddings from the CLIP model.
        ln_final (nn.Module): Layer normalization applied after the transformer.
        text_projection (Tensor): Linear projection layer for final text features.
        dtype (torch.dtype): Data type of the model, typically torch.FloatTensor.
    """

    def __init__(self, clip_model):
        """
        Initializes the TextEncoder module using components from a given CLIP model.

        Args:
            clip_model (CLIP): A pre-trained CLIP model.
        """
        super().__init__()
        # Adjust the CLIP model to the appropriate data type (float)
        clip_model = clip_model.type(torch.FloatTensor)

        # Extract relevant parts from the CLIP model
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Forward pass for encoding all text prompts.

        Args:
            prompts (Tensor): The context vectors for prompts of all classes.
            tokenized_prompts (Tensor): Tokenized representation of the prompts of all classes.

        Returns:
            Tensor: The encoded text features.
        """
        # Add positional embeddings to prompts and adjust dimensions for transformer
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # Reorder dimensions for transformer input. # (batch, length, dimension) -> (length, batch, dimension) for transformer

        # Pass the input through the transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Reorder dimensions back to original.  # (batch, length, dimension) <- (length, batch, dimension) for transformer

        # Apply layer normalization
        x = self.ln_final(x).type(self.dtype)

        # Extract features corresponding to the end-of-token (EOT) embedding
        # and apply text projection to get final text feature embeddings
        # EOT: embeddings of entire input sequence
        # self.text_projection is a learned linear transformation
        # maps the high-dimensional transformer output to a lower-dimensional space suitable for downstream tasks
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    
class MLP(nn.Module):
    """
    MLP to transform text_features to image features dimensions to use text encoder
    with any image encoder
    the temperature parameter is trained here and used for MM loss in the training loop
    output_dim is the dimension of the image_feature output of the image encoder
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Temperature as trainable param for mm loss
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x