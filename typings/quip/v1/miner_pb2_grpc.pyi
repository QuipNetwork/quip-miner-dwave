# Hand-written companion to the generated miner_pb2.pyi.
#
# protoc emits .pyi for messages (--pyi_out) but not for the gRPC service, and
# a stubPath package hides every sibling module it does not declare — without
# this file quip_solver_core's re-export of miner_pb2_grpc reads as missing.
#
# Only the surface this repo uses is declared. test_proto_stubs.py asserts these
# names exist on the installed module, so a pin bump that moves them fails a
# test instead of silently typing as Any.

from collections.abc import Iterable, Iterator

import grpc

from quip.v1 import miner_pb2

class MinerServiceStub:
    def __init__(self, channel: grpc.Channel) -> None: ...
    def Session(
        self, request_iterator: Iterable[miner_pb2.MinerMsg]
    ) -> Iterator[miner_pb2.CoordMsg]: ...

class MinerServiceServicer:
    def Session(
        self, request_iterator: Iterable[miner_pb2.MinerMsg], context: grpc.ServicerContext
    ) -> Iterator[miner_pb2.CoordMsg]: ...

def add_MinerServiceServicer_to_server(
    servicer: MinerServiceServicer, server: grpc.Server
) -> None: ...
