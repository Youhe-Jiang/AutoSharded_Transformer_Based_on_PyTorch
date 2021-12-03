import math

class MemCostModelPerLayer:
    def __init__(self, d_model, num_nodes, coder_type, fsdp_level):
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.coder_type = coder_type
        self.fsdp_level = fsdp_level

    def output_cost(self):
        if self.coder_type == 'encoder':
            cost = ((113.4 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / self.num_nodes
        else:
            cost = ((151.17 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / self.num_nodes
        return cost

memcost = MemCostModelPerLayer(d_model=1024, num_nodes=4, coder_type='encoder', fsdp_level=2).output_cost()


