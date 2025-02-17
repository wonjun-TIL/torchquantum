from operator import Operator

class Operation(Operator, metaclass=ABCMeta):
   """양자 게이트 연산을 위한 기본 클래스

   구체적인 양자 게이트들이 상속받아 사용할 기본 클래스입니다.
   파라미터 관리와 행렬 연산을 위한 기본 기능을 제공합니다.
   """
   def __init__(
       self,
       has_params: bool = False,
       trainable: bool = False,
       init_params=None,
       n_wires=None,
       wires=None,
       inverse=False,
   ):
       """Operation 클래스 초기화
       
       Args:
           has_params (bool, optional): 게이트가 파라미터를 가지는지 여부. 기본값은 False
           trainable (bool, optional): 파라미터가 학습 가능한지 여부. 기본값은 False
           init_params (torch.Tensor, optional): 초기 파라미터. 기본값은 None
           n_wires (int, optional): 큐비트 수. 기본값은 None
           wires (Union[int, List[int]], optional): 게이트가 적용될 큐비트들. 기본값은 None
           inverse (bool): 역연산 여부. 기본값은 False
       """
       super().__init__(
           has_params=has_params,
           trainable=trainable,
           init_params=init_params,
           n_wires=n_wires,
           wires=wires,
           inverse=inverse,
       )
       # num_wires가 정수인 경우 n_wires에 할당
       if type(self.num_wires) == int:
           self.n_wires = self.num_wires

   @property
   def matrix(self):
       """게이트의 유니타리 행렬 반환

       Returns:
           torch.Tensor: 게이트의 유니타리 행렬
       """
       op_matrix = self._matrix(self.params)
       return op_matrix

   @property
   def eigvals(self):
       """게이트 행렬의 고유값 반환

       Returns:
           torch.Tensor: 게이트 행렬의 고유값
       """
       op_eigvals = self._eigvals(self.params)
       return op_eigvals

   def init_params(self):
       """파라미터 초기화 메서드
       
       Raises:
           NotImplementedError: 하위 클래스에서 구현되어야 함
       """
       raise NotImplementedError

   def build_params(self, trainable):
       """파라미터 생성

       Args:
           trainable (bool): 파라미터의 학습 가능 여부

       Returns:
           torch.nn.Parameter: 생성된 파라미터
       """
       parameters = nn.Parameter(torch.empty([1, self.num_params], dtype=F_DTYPE))
       parameters.requires_grad = True if trainable else False
       return parameters

   def reset_params(self, init_params=None):
       """파라미터 재설정
       
       Args:
           init_params (Union[torch.Tensor, Iterable], optional): 초기화할 파라미터 값.
               리스트나 텐서로 제공가능. 기본값은 None
       """
       if init_params is not None:
           if isinstance(init_params, Iterable):
               for k, init_param in enumerate(init_params):
                   torch.nn.init.constant_(self.params[:, k], init_param)
           else:
               torch.nn.init.constant_(self.params, init_params)
       else:
           # 초기값이 주어지지 않은 경우 -π에서 π 사이의 균등분포로 초기화
           torch.nn.init.uniform_(self.params, -np.pi, np.pi)