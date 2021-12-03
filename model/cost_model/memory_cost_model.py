import math

class MemCostModelPerLayer:
    def __init__(self, d_model, num_nodes, coder_type, fsdp_level):
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.coder_type = coder_type
        self.fsdp_level = fsdp_level

    def output_cost(self):
        if self.num_nodes == 2:
            if self.coder_type == 'encoder':
                cost = ((126 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
            else:
                cost = ((168 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
        if self.num_nodes == 3:
            if self.coder_type == 'encoder':
                cost = ((117.67 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
            else:
                cost = ((156.75 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
        if self.num_nodes == 4:
            if self.coder_type == 'encoder':
                cost = ((113.42 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
            else:
                cost = ((151.17 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
        if self.num_nodes == 5:
            if self.coder_type == 'encoder':
                cost = ((110.92 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
            else:
                cost = ((147.67 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
        if self.num_nodes == 6:
            if self.coder_type == 'encoder':
                cost = ((109.17 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
            else:
                cost = ((145.5 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
        if self.num_nodes == 7:
            if self.coder_type == 'encoder':
                cost = ((108.08 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
            else:
                cost = ((143.92 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
        if self.num_nodes == 8:
            if self.coder_type == 'encoder':
                cost = ((107.17 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
            else:
                cost = ((142.75 - 33.5 * self.fsdp_level) * pow((self.d_model / 512), 2)) / 4
        return cost

