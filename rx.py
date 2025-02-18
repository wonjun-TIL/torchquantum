import torch
import torch.nn as nn
import numpy as np

class RXGate(nn.Module):
    """RX 게이트를 구현하는 클래스"""
    """
    RX 게이트는 X축을 중심으로 회전하는 단일 큐비트 게이트입니다.
    RX(θ) = exp(-iθX/2) = cos(θ/2)I - i sin(θ/2)X

    Args:
        n_qubits (int): 전체 큐비트 수
    """

    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        
    def get_rx_matrix(self, theta):
        """단일 큐비트 RX 행렬 생성"""
        cos = torch.cos(theta/2)
        sin = torch.sin(theta/2)
        return torch.tensor([[cos, -1j*sin],
                           [-1j*sin, cos]], dtype=torch.complex64)
    
    def forward(self, theta, target_qubit):
        """
        n_qubits: 전체 큐비트 수
        theta: 회전각
        target_qubit: RX 게이트를 적용할 큐비트 인덱스
        """
        if not 0 <= target_qubit < self.n_qubits:
            raise ValueError(f"Target qubit must be between 0 and {self.n_qubits-1}")
        
        # 단일 큐비트 RX 행렬
        rx = self.get_rx_matrix(theta)
        
        # Identity 행렬
        I = torch.eye(2, dtype=torch.complex64)
        
        # 전체 시스템 행렬 구성
        matrix = torch.tensor(1, dtype=torch.complex64)
        
        for i in range(self.n_qubits):
            if i == target_qubit:
                matrix = torch.kron(matrix, rx)
            else:
                matrix = torch.kron(matrix, I)
                
        return matrix

# 테스트
n_qubits = 2
multi_rx = RXGate(n_qubits)

# RX 게이트를 첫 번째 큐비트에 적용
theta = torch.tensor(np.pi/2)
matrix = multi_rx(theta, target_qubit=0)
print("2-qubit system RX(π/2) on first qubit:")
print(matrix)

# |00⟩ 상태에 적용
state = torch.zeros(2**n_qubits, dtype=torch.complex64)
state[0] = 1  # |00⟩ 상태 초기화
rotated_state = matrix @ state
print("\nInitial state |00⟩:", state)
print("After RX(π/2) on first qubit:", rotated_state)
