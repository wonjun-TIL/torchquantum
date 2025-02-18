import torch
import torch.nn as nn
import numpy as np


class RYGate(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        
    def get_ry_matrix(self, theta):
        """단일 큐비트 RY 행렬 생성"""
        cos = torch.cos(theta/2)
        sin = torch.sin(theta/2)
        return torch.tensor([[cos, -sin],
                           [sin, cos]], dtype=torch.complex64)
    
    def forward(self, theta, target_qubit):
        """
        n_qubits: 전체 큐비트 수
        theta: 회전각
        target_qubit: RY 게이트를 적용할 큐비트 인덱스
        """
        if not 0 <= target_qubit < self.n_qubits:
            raise ValueError(f"Target qubit must be between 0 and {self.n_qubits-1}")
        
        # 단일 큐비트 RY 행렬
        ry = self.get_ry_matrix(theta)
        
        # Identity 행렬
        I = torch.eye(2, dtype=torch.complex64)
        
        # 전체 시스템 행렬 구성
        matrix = torch.tensor(1, dtype=torch.complex64)
        
        for i in range(self.n_qubits):
            if i == target_qubit:
                matrix = torch.kron(matrix, ry)
            else:
                matrix = torch.kron(matrix, I)
                
        return matrix

# 테스트
def test_ry_gates():
    # 다중 큐비트 RY 테스트
    n_qubits = 2
    theta = torch.tensor(np.pi/2)  # 90도 회전
    multi_ry = RYGate(n_qubits)

    # RY 게이트를 첫 번째 큐비트에 적용
    matrix = multi_ry(theta, target_qubit=0)
    print("\n2-qubit system RY(π/2) on first qubit:")
    print(matrix)

    # |00⟩ 상태에 적용
    state = torch.zeros(2**n_qubits, dtype=torch.complex64)
    state[0] = 1  # |00⟩ 상태 초기화
    rotated_state = matrix @ state
    print("\nInitial state |00⟩:", state)
    print("After RY(π/2) on first qubit:", rotated_state)

if __name__ == "__main__":
    test_ry_gates()