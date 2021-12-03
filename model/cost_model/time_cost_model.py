import math
  
class TimeCostModelPerLayer:
    def __init__(self, d_model, coder_type, fsdp_level):
        self.d_model = d_model
        self.coder_type = coder_type
        self.fsdp_level = fsdp_level

    def output_cost(self):
        if self.coder_type == 'encoder':
            cost = (0.0013 + 0.006 * self.fsdp_level) * pow((self.d_model / 1024), 2)
        else:
            cost = (0.0024 + 0.006 * self.fsdp_level) * pow((self.d_model / 1024), 2)
        return cost
