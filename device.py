import torch
import torch.nn as nn

class QuantumDevice(nn.Module):
    def __init__(
        self,
        n_wires: int,
        bsz: int = 1,
        device: str = "cpu",
    ):
        """
        Args:
            n_wires: 큐비트의 수
            bsz: 배치 크기
            device: 연산 디바이스 ('cpu' 또는 'cuda')
        """
        super().__init__()
        self.n_wires = n_wires
        self.bsz = bsz
        self.device = device

        # 초기 상태 |0> 설정
        _state = torch.zeros(2**self.n_wires, dtype=torch.complex64)
        _state[0] = 1 + 0j
        _state = torch.reshape(_state, [2] * self.n_wires).to(self.device)
        self.register_buffer("state", _state)

        # 배치 처리를 위한 상태 확장
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self._states = self.state.repeat(*repeat_times)
        self.register_buffer("states", self._states)

    def set_states(self, states: torch.Tensor):
        """상태 설정"""
        bsz = states.shape[0]
        self.states = torch.reshape(states, [bsz] + [2] * self.n_wires)

    def reset_states(self, bsz: int):
        """상태 초기화"""
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = self.state.repeat(*repeat_times).to(self.state.device)

    def get_states_1d(self):
        """1차원 텐서 형태로 상태 반환"""
        bsz = self.states.shape[0]
        return torch.reshape(self.states, [bsz, 2**self.n_wires])