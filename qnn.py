import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import QuantumEncoder
from pqc import PQCLayer
from measurement import MeasurementLayer

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes=4):
        """
        Args:
            n_qubits (int): 사용할 큐비트의 수
            n_layers (int): PQC의 레이어 수
            n_classes (int): 분류할 클래스의 수
        """
        super().__init__()
        self.n_qubits = n_qubits
        
        # 1. 클래식 인코더
        self.encoder = QuantumEncoder(n_qubits)
        
        # 2. 양자 회로 레이어 (PQC)
        self.pqc = PQCLayer(n_qubits, n_layers)
        
        # 3. 측정 레이어
        self.measurement = MeasurementLayer(n_qubits)
        
        # 4. 최종 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        # 1. 이미지를 양자 회로 파라미터로 인코딩
        encoded_params = self.encoder(x)
        
        # 2. 초기 양자 상태 준비 (|0⟩^n)
        batch_size = x.shape[0]
        quantum_state = torch.zeros(batch_size, 2**self.n_qubits, dtype=torch.complex64)
        quantum_state[:, 0] = 1.0  # |00...0⟩ 상태
        
        # 3. PQC 적용
        quantum_state = self.pqc(quantum_state)
        
        # 4. 측정
        measurements = self.measurement(quantum_state)
        
        # 5. 분류
        output = self.classifier(measurements)
        
        return output

# 학습 함수
def train_qnn(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 순전파
            output = model(data)
            loss = criterion(output, target)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}: Average loss = {avg_loss:.6f}')

# 테스트 함수
def test_qnn():
    # 모델 파라미터
    n_qubits = 4
    n_layers = 2
    n_classes = 4
    
    # 모델 초기화
    model = QuantumNeuralNetwork(n_qubits, n_layers, n_classes)
    
    # 가상의 입력 데이터 생성
    batch_size = 5
    test_input = torch.randn(batch_size, 1, 28, 28)  # MNIST 형식
    
    # 모델 실행
    output = model(test_input)
    
    print("입력 shape:", test_input.shape)
    print("출력 shape:", output.shape)
    print("\n샘플 출력 (첫 번째 배치):")
    print(F.softmax(output[0], dim=0))

if __name__ == "__main__":
    test_qnn()