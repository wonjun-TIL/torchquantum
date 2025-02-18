import torch
import torch.nn as nn
import numpy as np

class CNOTGate(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
    
    def forward(self, control, target):
        """
        CNOT gate matrix:
        [1 0 0 0]
        [0 1 0 0]
        [0 0 0 1]
        [0 0 1 0]
        
        control: 제어 큐비트의 인덱스
        target: 목표 큐비트의 인덱스
        """
        if not (0 <= control < self.n_qubits and 0 <= target < self.n_qubits):
            raise ValueError(f"Qubit indices must be between 0 and {self.n_qubits-1}")
        if control == target:
            raise ValueError("Control and target qubits must be different")
            
        # 시스템 크기
        dim = 2**self.n_qubits
        matrix = torch.zeros((dim, dim), dtype=torch.complex64)
        
        # 모든 계산기저 상태에 대해
        for i in range(dim):
            # i를 이진수로 변환하여 각 큐비트의 상태 확인
            binary = format(i, f'0{self.n_qubits}b')
            # control 큐비트가 1이면 target 큐비트 반전
            if binary[control] == '1':
                # target 큐비트 반전
                output_binary = list(binary)
                output_binary[target] = '1' if binary[target] == '0' else '0'
                output_idx = int(''.join(output_binary), 2)
                matrix[output_idx, i] = 1
            else:
                # control 큐비트가 0이면 상태 그대로 유지
                matrix[i, i] = 1
                
        return matrix

# 테스트
def test_cnot_gate():
    n_qubits = 2
    cnot = CNOTGate(n_qubits)
    
    # control=0, target=1로 CNOT 게이트 생성
    matrix = cnot(control=0, target=1)
    print("CNOT matrix:")
    print(matrix)
    
    # 다양한 기저상태에 대해 테스트
    test_states = {
        "|00⟩": torch.tensor([1, 0, 0, 0], dtype=torch.complex64),
        "|01⟩": torch.tensor([0, 1, 0, 0], dtype=torch.complex64),
        "|10⟩": torch.tensor([0, 0, 1, 0], dtype=torch.complex64),
        "|11⟩": torch.tensor([0, 0, 0, 1], dtype=torch.complex64)
    }
    
    print("\nTesting CNOT on basis states:")
    for name, state in test_states.items():
        result = matrix @ state
        print(f"\nInput {name}:")
        print("Output:", result)

    # Bell 상태 생성 예제
    print("\nCreating Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:")
    # |+0⟩ 상태 준비 (첫 번째 큐비트에 Hadamard 적용)
    h_0 = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    i_1 = torch.eye(2, dtype=torch.complex64)
    h_tensor = torch.kron(h_0, i_1)
    
    # |00⟩ 초기 상태
    initial_state = torch.zeros(4, dtype=torch.complex64)
    initial_state[0] = 1
    
    # Hadamard 적용
    h_state = h_tensor @ initial_state
    
    # CNOT 적용
    bell_state = matrix @ h_state
    print("Result:", bell_state)

if __name__ == "__main__":
    test_cnot_gate()

# 양자 회로에서 자주 사용되는 유용한 함수들
def create_bell_state():
    """
    Bell 상태 |Φ+⟩ = (|00⟩ + |11⟩)/√2 생성
    """
    n_qubits = 2
    # 초기 상태 |00⟩
    state = torch.zeros(2**n_qubits, dtype=torch.complex64)
    state[0] = 1
    
    # Hadamard on first qubit
    h_0 = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    i_1 = torch.eye(2, dtype=torch.complex64)
    h_tensor = torch.kron(h_0, i_1)
    state = h_tensor @ state
    
    # CNOT
    cnot = CNOTGate(n_qubits)
    state = cnot(control=0, target=1) @ state
    
    return state