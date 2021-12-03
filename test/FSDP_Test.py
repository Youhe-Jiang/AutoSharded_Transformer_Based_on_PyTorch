from contextlib import suppress
from enum import Enum
import os
import sys
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    get_cycles_per_ms,
)

class FSDPTest(MultiProcessTestCase):
    
    def setUp(self):
        super(FSDPTest, self).setUp()
        self._spawn_processes()
    
    @property
    def world_size(self):
        return torch.cuda.device_count() if torch.cuda.is_available() else 4

    @property
    def init_method(self):
        return "{}{file_name}".format(FILE_SCHEMA, file_name=self.file_name)

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        # Specify gloo backend to make 'init_process_group()' succeed,
        # Actual tests will be skipped if there is no enough GPUs.

        backend = os.environ.get("BACKEND", None)
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        try:
            dist.init_process_group(
                init_method=self.init_method,
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        if torch.cuda.is_available() and torch.cuda.device_count():
            torch.cuda.set_device(self.rank % torch.cuda.device_count())

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier()

        self.run_test(test_name, pipe)

        dist.barrier()

        dist.destroy_process_group()
        sys.exit(0)
