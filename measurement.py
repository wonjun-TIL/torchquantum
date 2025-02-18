import torch
import torch.nn as nn
import numpy as np

class MeasurementLayer(nn.Module):
    def __init__(self, n_qubits):
        """
        Args:
            n_qubits (int): 측정할 큐비트의 수
        """
        super().__init__()
        self.n_qubits = n_qubits
        
        # Pauli Z 연산자 초기화
        self.pauli_z = torch.tensor([[1, 0], 
                                    [0, -1]], dtype=torch.complex64)
        
    def get_measurement_operator(self, target_qubit):
        """특정 큐비트에 대한 측정 연산자 생성"""
        # Identity 행렬
        I = torch.eye(2, dtype=torch.complex64)
        
        # 전체 시스템에 대한 측정 연산자 구성
        operator = torch.tensor(1, dtype=torch.complex64)
        
        for i in range(self.n_qubits):
            if i == target_qubit:
                operator = torch.kron(operator, self.pauli_z)
            else:
                operator = torch.kron(operator, I)
                
        return operator
    
    def forward(self, quantum_state):
        """
        Args:
            quantum_state: 측정할 양자 상태 (2^n_qubits 차원의 벡터)
        Returns:
            measurements: 각 큐비트의 기대값
        """
        # 배치 처리를 위한 차원 확인
        batch_size = quantum_state.shape[0] if len(quantum_state.shape) > 1 else 1
        if len(quantum_state.shape) == 1:
            quantum_state = quantum_state.unsqueeze(0)
            
        measurements = []
        
        # 각 큐비트에 대해 측정
        for qubit in range(self.n_qubits):
            # 측정 연산자 생성
            operator = self.get_measurement_operator(qubit)
            
            # 기대값 계산: <ψ|Z|ψ>
            expectation = torch.real(
                torch.einsum('bi,ij,bj->b', 
                           quantum_state.conj(), 
                           operator, 
                           quantum_state)
            )
            measurements.append(expectation)
            
        # 측정 결과를 텐서로 변환 [batch_size, n_qubits]
        measurements = torch.stack(measurements, dim=1)
        
        return measurements.squeeze()

# 테스트
def test_measurement():
    n_qubits = 2
    measurement_layer = MeasurementLayer(n_qubits)
    
    # 테스트 케이스 1: |0⟩ 상태
    state_0 = torch.zeros(2**n_qubits, dtype=torch.complex64)
    state_0[0] = 1.0
    print("측정 테스트 1 - |0⟩ 상태:")
    print(measurement_layer(state_0))  # 모든 큐비트가 +1을 측정값으로 가져야 함
    
    # 테스트 케이스 2: |1⟩ 상태
    state_1 = torch.zeros(2**n_qubits, dtype=torch.complex64)
    state_1[3] = 1.0  # |11⟩ 상태
    print("\n측정 테스트 2 - |11⟩ 상태:")
    print(measurement_layer(state_1))  # 모든 큐비트가 -1을 측정값으로 가져야 함
    
    # 테스트 케이스 3: 중첩 상태
    state_plus = torch.ones(2**n_qubits, dtype=torch.complex64) / np.sqrt(2**n_qubits)
    print("\n측정 테스트 3 - 균등 중첩 상태:")
    print(measurement_layer(state_plus))  # 모든 큐비트가 0에 가까운 값을 가져야 함
    
    # 테스트 케이스 4: 배치 처리
    batch_size = 3
    batch_states = torch.stack([state_0, state_1, state_plus])
    print("\n측정 테스트 4 - 배치 처리:")
    print(measurement_layer(batch_states))

if __name__ == "__main__":
    test_measurement()