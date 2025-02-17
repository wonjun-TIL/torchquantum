import torch
import torch.nn as nn

class QuantumModule(nn.Module):
    """양자 연산을 수행하기 위한 기본 모듈
    
    PyTorch의 nn.Module을 상속받아 양자 연산에 특화된 기능을 제공합니다.
    이 클래스는 양자 회로의 기본 구성 요소로 사용됩니다.

    Attributes:
        n_wires (int): 양자 회로에서 사용할 큐비트의 수
        q_device (tq.QuantumDevice): 양자 상태를 저장하고 조작하는 디바이스 객체
        device (torch.device): 연산이 실행될 디바이스 (CPU 또는 GPU)
    """

    def __init__(self, n_wires: int) -> None:
        """QuantumModule 초기화
        
        Args:
            n_wires (int): 양자 회로에서 사용할 큐비트의 수
        
        Example:
            >>> qmodule = QuantumModule(n_wires=2)
        """
        super().__init__()
        self.n_wires = n_wires
        self.q_device = None  # 양자 상태를 다루는 디바이스
        self.device = None    # CPU/GPU 디바이스

    def forward(self, q_device):
        """순전파 연산을 수행
        
        이 메서드는 각 하위 클래스에서 구체적인 양자 연산을 구현할 때 오버라이드됩니다.
        
        Args:
            q_device: 양자 상태를 저장하고 있는 디바이스 객체
                     양자 상태에 대한 게이트 연산을 수행하는 데 사용됩니다.
        
        Raises:
            NotImplementedError: 이 메서드는 하위 클래스에서 구현되어야 합니다.
        
        Example:
            >>> class MyQuantumGate(QuantumModule):
            ...     def forward(self, q_device):
            ...         # 구체적인 양자 게이트 연산 구현
            ...         pass
        """
        raise NotImplementedError("Forward 메서드는 하위 클래스에서 구현되어야 합니다.")

