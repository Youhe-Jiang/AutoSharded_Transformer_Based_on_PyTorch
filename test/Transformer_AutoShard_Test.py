import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from torch.optim import Adam
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_full_params,
)
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, run_tests
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper, disable_checkpointing
import time
from Transformer_AutoShard_Layer import TransformerEncoderLayer, TransformerDecoderLayer
from AutoShardFunction import AutoShardFunction 
from Transformer_AutoShard import Transformer_AutoShard

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

class TestUnevenParamShard(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_one_iteration(self):
        model = Transformer_AutoShard(
                d_model=1536,
                nhead=48,
                num_nodes=self.world_size,
                num_encoder_layers=24,
                num_decoder_layers=24,
                max_memory=10000,
                )        
        my_lr = 6e-4
        model = checkpoint_wrapper(model, offload_to_cpu=True)
        model = FSDP(model.cuda())
        print_peak_memory("Peak mem after load model", self.rank)
        optim = Adam(model.parameters(), lr=my_lr)
        src = torch.rand((10, 32, 1536))
        tgt = torch.rand((20, 32, 1536))
        print_peak_memory("Peak mem before forward", self.rank)
        for i in range(10):
                torch.cuda.synchronize()
                start = time.time()
                out = model(src.cuda(), tgt.cuda())
                torch.cuda.synchronize()
                end = time.time()
                print(end-start)
                print_peak_memory("Peak mem after forward", self.rank)

if __name__ == "__main__":
    run_tests()
