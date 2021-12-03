from cost_model.memory_cost_model import MemCostModelPerLayer 
from cost_model.time_cost_model import TimeCostModelPerLayer

class AutoShardFunction:
    def __init__(self, d_model, num_nodes, num_encoder_layers, num_decoder_layers, max_memory):
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_memory = max_memory

    def layer_cost(self):
        encoder_layer_cost = []
        decoder_layer_cost = []
        encoder_layer_time = []
        decoder_layer_time = []
        for i in range(4):
            encoder_layer_cost.append(
                    MemCostModelPerLayer(d_model=self.d_model, num_nodes=self.num_nodes, coder_type='encoder', fsdp_level=i).output_cost(),
            )
            encoder_layer_time.append(
                    TimeCostModelPerLayer(d_model=self.d_model, coder_type='encoder', fsdp_level=i).output_cost(),
            )
        for i in range(5):
            decoder_layer_cost.append(
                    MemCostModelPerLayer(d_model=self.d_model, num_nodes=self.num_nodes, coder_type='decoder', fsdp_level=i).output_cost(),
            )
            decoder_layer_time.append(
                    TimeCostModelPerLayer(d_model=self.d_model, coder_type='decoder', fsdp_level=i).output_cost(),
            )
        return encoder_layer_cost, decoder_layer_cost, encoder_layer_time, decoder_layer_time

    def auto_shard(self):
        encoder_layer_cost, decoder_layer_cost, encoder_layer_time, decoder_layer_time = self.layer_cost()
        encoder_layer_batches = int(self.num_encoder_layers / 6)
        decoder_layer_batches = int(self.num_decoder_layers / 6)
        index = []
        memory_cost = []
        time_cost = []
        memory_encoder = [i * encoder_layer_batches for i in encoder_layer_cost]
        memory_decoder = [i * decoder_layer_batches for i in decoder_layer_cost]
        time_encoder = [i * encoder_layer_batches for i in encoder_layer_time]
        time_decoder = [i * decoder_layer_batches for i in decoder_layer_time]
        for a in range(len(encoder_layer_cost)):
            for b in range(len(decoder_layer_cost)):
                for c in range(len(encoder_layer_cost)):
                    for d in range(len(decoder_layer_cost)):
                        for e in range(len(encoder_layer_cost)):
                            for f in range(len(decoder_layer_cost)):
                                for g in range(len(encoder_layer_cost)):
                                    for h in range(len(decoder_layer_cost)):
                                        for i in range(len(encoder_layer_cost)):
                                            for j in range(len(decoder_layer_cost)):
                                                for k in range(len(encoder_layer_cost)):
                                                    for l in range(len(decoder_layer_cost)):
                                                        memory_total = memory_encoder[a] + memory_decoder[b] + memory_encoder[c] + memory_decoder[d] + memory_encoder[e] + memory_decoder[f] + memory_encoder[g] + memory_decoder[h] + memory_encoder[i] + memory_decoder[j] + memory_encoder[k] + memory_decoder[l]
                                                        time_total = time_encoder[a] + time_decoder[b] + time_encoder[c] + time_decoder[d] + time_encoder[e] + time_decoder[f] + time_encoder[g] + time_decoder[h] + time_encoder[i] + time_decoder[j] + time_encoder[k] + time_decoder[l]
                                                        if (memory_total < self.max_memory):
                                                            if not memory_total in memory_cost:
                                                                memory_cost.append(memory_total)
                                                                time_cost.append(time_total)
                                                                index.append([a, b, c, d, e, f, g, h, i, j, k, l])
        index = index[time_cost.index(min(time_cost))] 
        encoder_idx = []
        decoder_idx = []
        for i in range(len(index)):
            if i % 2 == 0:
                encoder_idx.append(index[i])
            else:
                decoder_idx.append(index[i])
        memory_cost = memory_cost[time_cost.index(min(time_cost))]
        return encoder_idx, decoder_idx
