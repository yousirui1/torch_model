import numpy as np
import torch
import re

class GlobalCMVN(torch.nn.Module):

    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer('mean', mean)
        self.register_buffer('istd', istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


def load_cmvn(cmvn_file):
    """ Load the kaldi format cmvn stats file and no need to calculate

    Args:
        cmvn_file: cmvn stats file in kaldi format

    Returns:
        a numpy array of [means, vars]
    """

    means = None
    variance = None
    with open(cmvn_file) as f:
        all_lines = f.readlines()
        for idx, line in enumerate(all_lines):
            if line.find('AddShift') != -1: 
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                means_str = re.findall(r'[\[](.*?)[\]]', next_line)[0]
                means_list = means_str.strip().split(' ')
                means = [0 - float(s) for s in means_list]
                assert len(means) == int(segs[1])
            elif line.find('Rescale') != -1: 
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                vars_str = re.findall(r'[\[](.*?)[\]]', next_line)[0]
                vars_list = vars_str.strip().split(' ')
                variance = [float(s) for s in vars_list]
                assert len(variance) == int(segs[1])
            elif line.find('Splice') != -1: 
                segs = line.strip().split(' ')
                assert len(segs) == 3
                next_line = all_lines[idx + 1]
                splice_str = re.findall(r'[\[](.*?)[\]]', next_line)[0]
                splice_list = splice_str.strip().split(' ')
                assert len(splice_list) * int(segs[2]) == int(segs[1])
                copy_times = len(splice_list)
            else:
                continue

    cmvn = np.array([means, variance])
    cmvn = np.tile(cmvn, (1, copy_times))

    return cmvn
