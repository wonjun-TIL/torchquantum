import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumEncoder(nn.Module):
    def __init__(self, n_qubits):
        """
        Args:
            n_qubits (int): 사용할 큐비트의 수 (인코딩의 출력 차원)
        """
        super().__init__()
        self.n_qubits = n_qubits
        
        # CNN 기반 인코더
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        
        # 이미지 크기에 따른 flatten 차원 계산
        # MNIST 기준 (28x28) -> 14x14 -> 7x7
        self.fc_input_dim = 32 * 7 * 7
        
        # 최종 양자 상태를 위한 선형 레이어
        self.fc = nn.Linear(self.fc_input_dim, n_qubits)
        
    def forward(self, x):
        # CNN 특징 추출
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.dropout(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_dim)
        
        # 양자 상태로 인코딩
        x = self.fc(x)
        
        # 각도로 정규화 (-π에서 π 사이)
        x = torch.tanh(x) * torch.pi
        
        return x

# 테스트
def test_encoder():
    # 테스트 파라미터
    batch_size = 5
    n_qubits = 4
    
    # 모델 초기화
    encoder = QuantumEncoder(n_qubits=n_qubits)
    
    # 가상의 입력 데이터 생성 (MNIST 형식)
    test_input = torch.randn(batch_size, 1, 28, 28)
    
    # 인코딩
    encoded = encoder(test_input)
    
    print("입력 데이터 shape:", test_input.shape)
    print("인코딩된 데이터 shape:", encoded.shape)
    print("\n샘플 인코딩 결과 (첫 번째 이미지):")
    print(encoded[0])
    print("\n각도 범위 확인:")
    print("최소값:", encoded.min().item())
    print("최대값:", encoded.max().item())

if __name__ == "__main__":
    test_encoder()