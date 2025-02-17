from ..operator import Operation
from abc import ABCMeta
import funtional as tqf

class RX(Operation, metaclass=ABCMeta):
    """X축 회전 게이트를 위한 클래스
    
    단일 큐비트에 대해 X축을 중심으로 회전 연산을 수행하는 양자 게이트입니다.
    
    Attributes:
        num_params (int): 게이트의 파라미터 개수 (회전각)
        num_wires (int): 게이트가 작용하는 큐비트 수
        op_name (str): 게이트의 이름
        func (staticmethod): 실제 게이트 연산을 수행하는 함수
    """
    num_params = 1   # 회전각 하나만 필요
    num_wires = 1    # 단일 큐비트 게이트
    op_name = "rx"   # 게이트 이름
    func = staticmethod(tqf.rx)  # 실제 연산 함수

    @classmethod
    def _matrix(cls, params):
        """게이트의 유니타리 행렬을 반환
        
        Args:
            params: 회전각 파라미터
            
        Returns:
            torch.Tensor: RX 게이트의 유니타리 행렬
        """
        return tqf.rx_matrix(params)