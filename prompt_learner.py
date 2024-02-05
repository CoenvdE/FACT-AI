import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


# prompt learner as implemented in the LICO and COOP code.
class PromptLearner(nn.Module):
    """
    A PyTorch module for learning prompt embeddings in the context of a CLIP model.

    This module creates and learns context vectors (prompts) for each class in a given set of class names.
    These prompts are used with a CLIP model to produce text embeddings that are aligned with image features.

    Attributes:
        ctx (nn.Parameter): Learnable context vectors for each class.
        token_prefix (Tensor): Start-of-sequence token embeddings from CLIP.
        token_suffix (Tensor): End-of-sequence and class token embeddings from CLIP.
        n_cls (int): Number of classes.
        n_ctx (int): Number of context tokens.
        class_token_position (str): Position of the class token in the prompt (options: 'middle', 'end', 'front').
    """
    def __init__(self, classnames, clip_model):
        """
        Initializes the PromptLearner module with class names and a CLIP model.

        Args:
            classnames (list): A list of class names (strings).
            clip_model (CLIP): The pre-trained CLIP model from which certain layers are used.
        """
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 0
        ctx_init = None
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.N = 1

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if True:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        # '.' as end of sentence token for representation of whole sentence
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # (10, 77)
        tokenized_prompts = tokenized_prompts.repeat(self.N,1)


        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        print('tokenized prompts:', embedding.shape, 'ctx: ', self.ctx.shape)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'middle'

    # DAAN: Why do we give the prefix and suffix here if we dont use them?
    def _ctx_shuffle(self, prefix, suffix, ctx, cls_loc = 'end', shuffleCLS = False):
        """
        Shuffles the context vectors.

        Args:
            prefix (Tensor): Prefix token embeddings.
            suffix (Tensor): Suffix token embeddings.
            ctx (Tensor): Context vectors to shuffle.
            cls_loc (str): Position of the class token in the prompt.
            shuffleCLS (bool): Whether to shuffle the class token positions.

        Returns:
            Tensor: Shuffled context vectors.
        """

        # shuffle the ctx along 2nd dimension
        rand_idx = torch.randperm(ctx.shape[1])
        shuffled_ctx = ctx[:, rand_idx, :]
        return shuffled_ctx


    def forward(self):
        """
        Forward pass of the PromptLearner to create prompts for each class.

        Returns:
            Tensor: A batch of prompts, one for each class.
        """

        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0)

        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = self._ctx_shuffle(prefix, suffix, ctx)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        return prompts