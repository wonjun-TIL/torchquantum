import torch
import torch.nn as nn
import numpy as np

class RZGate(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        
    def get_rz_matrix(self, theta):
        """단일 큐비트 RZ 행렬 생성"""
        neg_phase = torch.exp(-1j * theta/2)
        pos_phase = torch.exp(1j * theta/2)
        return torch.tensor([[neg_phase, 0],
                           [0, pos_phase]], dtype=torch.complex64)
    
    def forward(self, theta, target_qubit):
        """
        n_qubits: 전체 큐비트 수
        theta: 회전각
        target_qubit: RZ 게이트를 적용할 큐비트 인덱스
        """
        if not 0 <= target_qubit < self.n_qubits:
            raise ValueError(f"Target qubit must be between 0 and {self.n_qubits-1}")
        
        # 단일 큐비트 RZ 행렬
        rz = self.get_rz_matrix(theta)
        
        # Identity 행렬
        I = torch.eye(2, dtype=torch.complex64)
        
        # 전체 시스템 행렬 구성
        matrix = torch.tensor(1, dtype=torch.complex64)
        
        for i in range(self.n_qubits):
            if i == target_qubit:
                matrix = torch.kron(matrix, rz)
            else:
                matrix = torch.kron(matrix, I)
                
        return matrix

# 테스트
def test_rz_gates():
    # 다중 큐비트 RZ 테스트
    n_qubits = 2
    theta = torch.tensor(np.pi/2)  # 90도 회전
    multi_rz = RZGate(n_qubits)

    # RZ 게이트를 첫 번째 큐비트에 적용
    matrix = multi_rz(theta, target_qubit=0)
    print("\n2-qubit system RZ(π/2) on first qubit:")
    print(matrix)

    # |++⟩ 상태 준비 (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
    state = torch.ones(2**n_qubits, dtype=torch.complex64) / 2
    rotated_state = matrix @ state
    print("\nInitial state |++⟩:", state)
    print("After RZ(π/2) on first qubit:", rotated_state)

if __name__ == "__main__":
    test_rz_gates()