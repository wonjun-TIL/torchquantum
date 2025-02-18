import torch
import torch.nn as nn
import numpy as np

from rx import RXGate
from ry import RYGate
from rz import RZGate
from cnot import CNOTGate




class PQCLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        """
        Args:
            n_qubits (int): 큐비트의 수
            n_layers (int): PQC의 반복 레이어 수
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 각 레이어의 각 큐비트에 대한 회전 파라미터
        self.rotation_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3)  # (레이어, 큐비트, 회전축)
        )
        
        # MultiQubit 게이트 초기화
        self.rx = RXGate(n_qubits)
        self.ry = RYGate(n_qubits)
        self.rz = RZGate(n_qubits)
        self.cnot = CNOTGate(n_qubits)
        
    def apply_rotation_layer(self, state, layer_idx):
        """한 레이어의 회전 게이트들 적용"""
        batch_size = state.shape[0]
        
        for qubit in range(self.n_qubits):
            # 각 회전 파라미터 가져오기
            theta_x = self.rotation_params[layer_idx, qubit, 0]
            theta_y = self.rotation_params[layer_idx, qubit, 1]
            theta_z = self.rotation_params[layer_idx, qubit, 2]
            
            # MultiQubit 회전 게이트 적용
            rx_matrix = self.rx(theta_x, target_qubit=qubit)
            ry_matrix = self.ry(theta_y, target_qubit=qubit)
            rz_matrix = self.rz(theta_z, target_qubit=qubit)
            
            # 배치 차원을 고려한 행렬 곱셈
            state = torch.bmm(
                rx_matrix.expand(batch_size, -1, -1),
                state.unsqueeze(-1)
            ).squeeze(-1)
            
            state = torch.bmm(
                ry_matrix.expand(batch_size, -1, -1),
                state.unsqueeze(-1)
            ).squeeze(-1)
            
            state = torch.bmm(
                rz_matrix.expand(batch_size, -1, -1),
                state.unsqueeze(-1)
            ).squeeze(-1)
            
        return state

    def apply_entanglement_layer(self, state):
        """CNOT 게이트로 얽힘 생성"""
        batch_size = state.shape[0]
        
        for i in range(0, self.n_qubits-1):
            cnot_matrix = self.cnot(control=i, target=i+1)
            state = torch.bmm(
                cnot_matrix.expand(batch_size, -1, -1),
                state.unsqueeze(-1)
            ).squeeze(-1)
        
        return state
            
    def forward(self, state):
        """
        Args:
            state: 초기 양자 상태 (2^n_qubits 차원의 벡터)
        Returns:
            state: 변환된 양자 상태
        """
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        
        # 각 레이어 순차적으로 적용
        for layer in range(self.n_layers):
            # 1. 회전 게이트들 적용
            state = self.apply_rotation_layer(state, layer)
            
            # 2. 얽힘 레이어 적용
            state = self.apply_entanglement_layer(state)
            
        return state

# 테스트 함수
def test_pqc():
    # 테스트 파라미터
    n_qubits = 2
    n_layers = 2
    
    # PQC 레이어 초기화
    pqc = PQCLayer(n_qubits=n_qubits, n_layers=n_layers)
    
    # 초기 상태 |00⟩ 준비
    initial_state = torch.zeros(2**n_qubits, dtype=torch.complex64)
    initial_state[0] = 1.0
    
    # PQC 적용
    final_state = pqc(initial_state)
    
    print("초기 상태:", initial_state)
    print("\n최종 상태:")
    print(final_state)
    
    # 상태 벡터의 정규화 확인
    norm = torch.abs(torch.sum(torch.abs(final_state)**2))
    print("\n상태 벡터 norm (should be close to 1):", norm.item())

if __name__ == "__main__":
    test_pqc()