
from typing import Any, List, Tuple

import torch

class ScorerInterface:
    """
    Scorer interface for beam search

    The scorer performs scoring of the all tokens in vocabulary.

    Examples:
        * Search heuristics
            * :class:`espnet.nets.scorers.length_bonus.LengthBonus`
        * Decoder networks of the sequence-to-sequence models
            * :class:`espnet.nets.pytorch_backend.nets.transformer.decoder.Decoder`
            * :class:`espnet.nets.pytorch_backend.nets.rnn.decoders.Decoder`
        * Neural language models
            * :class:`espnet.nets.pytorch_backend.lm.transformer.TransformerLM`
            * :class:`espnet.nets.pytorch_backend.lm.default.DefaultRNNLM`
            * :class:`espnet.nets.pytorch_backend.lm.seq_rnn.SequentialRNNLM`
    """
    def init_state(self, x: torch.Tensor) -> Any:
        """
        Get an initial state for decoding (optional).
        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state
        """
        return None

    def select_state(self, state: Any, i: int, new_id: int = None)->Any:
        """
        Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i(int): Index to select a state in the main beam search
            new_id(int): New label index to select a state if necessary

        Returns:
            state: pruned state
        """
        return None if state is None else state[i]

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor)->Tuple[torch.Tensor, Any]:
        """
        Score new token (required).
        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): The encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                scores for next token that has a shape of `(n_vocab)`
                and next state for ys
        """
        raise NotImplementedError

    def final_score(self, state: Any) -> float:
        """
        Score eos (optional).
        Args:
            state: Scorer state for prefix tokens
        Returns:
            float: final score
        """
        return 0.0

class BatchScorerInterface(ScorerInterface):
    """ Batch scorer interface. """

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """
        Get an initial state for decoding (optional).
        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return self.init_state(x)

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
        Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs(torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        #""
        scores = list()
        outstates = list()
        for i, (y, state, x) in enumerate(zip(ys, states, xs)):
            score, outstate = self.score(y, state, x)
            outstates.append(outstate)
            scores.append(score)
        scores = torch.cat(scores, 0).view(ys.shape[0], -1)
        return scores, outstates

