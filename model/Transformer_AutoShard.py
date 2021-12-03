import torch.nn as nn
from Transformer_AutoShard_Layer import TransformerEncoderLayer, TransformerDecoderLayer
from AutoShardFunction import AutoShardFunction 

class Transformer_AutoShard(nn.Module):
    def __init__(self, d_model, nhead, num_nodes, num_encoder_layers, num_decoder_layers, max_memory):
        super().__init__()
        encoder_fsdp_level, decoder_fsdp_level = AutoShardFunction(d_model, num_nodes, num_encoder_layers, num_decoder_layers, max_memory).auto_shard()
        self.encoder = nn.Sequential(
            *[TransformerEncoderLayer(d_model, nhead, fsdp_level=encoder_fsdp_level[0]) for _ in range(int(num_encoder_layers/6))],
            *[TransformerEncoderLayer(d_model, nhead, fsdp_level=encoder_fsdp_level[1]) for _ in range(int(num_encoder_layers/6))],
            *[TransformerEncoderLayer(d_model, nhead, fsdp_level=encoder_fsdp_level[2]) for _ in range(int(num_encoder_layers/6))],
            *[TransformerEncoderLayer(d_model, nhead, fsdp_level=encoder_fsdp_level[3]) for _ in range(int(num_encoder_layers/6))],
            *[TransformerEncoderLayer(d_model, nhead, fsdp_level=encoder_fsdp_level[4]) for _ in range(int(num_encoder_layers/6))],
            *[TransformerEncoderLayer(d_model, nhead, fsdp_level=encoder_fsdp_level[5]) for _ in range(int(num_encoder_layers/6))],
            nn.LayerNorm(d_model),
        )
        self.decoder = nn.Sequential(
            *[TransformerDecoderLayer(d_model, nhead, fsdp_level=decoder_fsdp_level[0]) for _ in range(int(num_decoder_layers/6))],
            *[TransformerDecoderLayer(d_model, nhead, fsdp_level=decoder_fsdp_level[1]) for _ in range(int(num_decoder_layers/6))],
            *[TransformerDecoderLayer(d_model, nhead, fsdp_level=decoder_fsdp_level[2]) for _ in range(int(num_decoder_layers/6))],
            *[TransformerDecoderLayer(d_model, nhead, fsdp_level=decoder_fsdp_level[3]) for _ in range(int(num_decoder_layers/6))],
            *[TransformerDecoderLayer(d_model, nhead, fsdp_level=decoder_fsdp_level[4]) for _ in range(int(num_decoder_layers/6))],
            *[TransformerDecoderLayer(d_model, nhead, fsdp_level=decoder_fsdp_level[5]) for _ in range(int(num_decoder_layers/6))],
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, src, tgt):
        src = self.encoder(src)
        input = (tgt, src)
        tgt = self.decoder(input)[0]
        tgt = self.layernorm(tgt)
        return tgt

