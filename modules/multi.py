

class MultiModule(nn.Module):
    """
    Displacement + Pruning を同時に計算
    """

    def __init__(self, cfgs):
        super().__init__()
        self.disp = DisplacementModule(cfgs)
        self.prune = PruningModule(cfgs)

    def forward(self, F_prime, d_prime, s_prime):
        delta = self.disp(F_prime, s_prime)
        p = self.prune(F_prime, d_prime, s_prime)
        return delta, p
