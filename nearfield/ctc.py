import torch
from collections import defaultdict
from typing import List, Optional, Tuple

def ctc_prefix_beam_search(
    logits: torch.Tensor,
    #logits_lengths: torch.Tensor,
    keywords_tokenset: set = None,
    score_beam_size: int = 3,
    path_beam_size: int = 20, 
) -> Tuple[List[List[int]], torch.Tensor]:
    """ CTC prefix beam search inner implementation

    Args:
        logits (torch.Tensor): (1, max_len, vocab_size)
        logits_lengths (torch.Tensor): (1, )
        keywords_tokenset (set): token set for filtering score
        score_beam_size (int): beam size for score
        path_beam_size (int): beam size for path

    Returns:
        List[List[int]]: nbest results
    """
    maxlen = logits.size(0)
    # ctc_probs = logits.softmax(1)  # (1, maxlen, vocab_size)
    ctc_probs = logits

    cur_hyps = [(tuple(), (1.0, 0.0, []))]

    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        probs = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (0.0, 0.0, []))

        # 2.1 First beam prune: select topk best
        top_k_probs, top_k_index = probs.topk(
            score_beam_size)  # (score_beam_size,)

        # filter prob score that is too small
        filter_probs = []
        filter_index = []
        for prob, idx in zip(top_k_probs.tolist(), top_k_index.tolist()):
            if keywords_tokenset is not None:
                if prob > 0.05 and idx in keywords_tokenset:
                    filter_probs.append(prob)
                    filter_index.append(idx)
            else:
                if prob > 0.05:
                    filter_probs.append(prob)
                    filter_index.append(idx)

        if len(filter_index) == 0:
            continue

        for s in filter_index:
            ps = probs[s].item()

            for prefix, (pb, pnb, cur_nodes) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb, nodes = next_hyps[prefix]
                    n_pb = n_pb + pb * ps + pnb * ps
                    nodes = cur_nodes.copy()
                    next_hyps[prefix] = (n_pb, n_pnb, nodes)
                elif s == last:
                    if not math.isclose(pnb, 0.0, abs_tol=0.000001):
                        # Update *ss -> *s;
                        n_pb, n_pnb, nodes = next_hyps[prefix]
                        n_pnb = n_pnb + pnb * ps
                        nodes = cur_nodes.copy()
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            nodes[-1]['prob'] = ps
                            nodes[-1]['frame'] = t
                        next_hyps[prefix] = (n_pb, n_pnb, nodes)

                    if not math.isclose(pb, 0.0, abs_tol=0.000001):
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb, nodes = next_hyps[n_prefix]
                        n_pnb = n_pnb + pb * ps
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                        next_hyps[n_prefix] = (n_pb, n_pnb, nodes)
                else:
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb, nodes = next_hyps[n_prefix]
                    if nodes:
                        if ps > nodes[-1]['prob']:  # update frame and prob
                            nodes[-1]['prob'] = ps
                            nodes[-1]['frame'] = t
                    else:
                        nodes = cur_nodes.copy()
                        nodes.append(dict(token=s, frame=t,
                                          prob=ps))  # to record token prob
                    n_pnb = n_pnb + pb * ps + pnb * ps
                    next_hyps[n_prefix] = (n_pb, n_pnb, nodes)

        # 2.2 Second beam prune
        next_hyps = sorted(
            next_hyps.items(), key=lambda x: (x[1][0] + x[1][1]), reverse=True)

        cur_hyps = next_hyps[:path_beam_size]

    hyps = [(y[0], y[1][0] + y[1][1], y[1][2]) for y in cur_hyps]
    return hyps
