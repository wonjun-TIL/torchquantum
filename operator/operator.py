


class Operator(QuantumModule):
    """양자 연산자들의 기본 클래스"""

    def __init__(
        self,
        has_params: bool = False,      # 파라미터를 가지는지 여부
        trainable: bool = False,       # 파라미터가 학습 가능한지 여부
        init_params=None,              # 초기 파라미터
        n_wires=None,                  # 큐비트 수 (MultiRZ와 같이 임의의 큐비트에 적용 가능한 게이트용)
        wires=None,                    # 연산자가 적용될 특정 큐비트
        inverse=False,                 # 역연산 여부
    ):
        super().__init__()
        self.params = None             # 연산자의 파라미터
        self.n_wires = n_wires        # 큐비트 수 저장
        self.wires = wires            # 적용될 큐비트 저장
        self._name = self.__class__.__name__  # 클래스 이름을 연산자 이름으로 사용
        self.inverse = inverse         # 역연산 여부 저장

        # 학습 가능한 경우 반드시 파라미터가 있어야 함
        assert not (trainable and not has_params), "학습 가능한 모듈은 반드시 파라미터를 가져야 합니다."

        self.has_params = has_params   # 파라미터 보유 여부 저장
        self.trainable = trainable     # 학습 가능 여부 저장

        # 파라미터가 있는 경우 초기화
        if self.has_params:
            self.params = self.build_params(trainable=self.trainable)
            self.reset_params(init_params)

    @property
    def name(self):
        """연산자의 이름 반환"""
        return self._name

    @name.setter
    def name(self, value):
        """연산자의 이름 설정"""
        self._name = value

    @classmethod
    def _matrix(cls, params):
        """연산자의 유니타리 행렬 반환
        각 게이트 클래스에서 구체적으로 구현해야 함"""
        raise NotImplementedError

    @property
    def matrix(self):
        """현재 파라미터로 유니타리 행렬 계산"""
        return self._matrix(self.params)

    def set_wires(self, wires):
        """연산자가 적용될 큐비트 설정
        
        Args:
            wires: 단일 큐비트(정수) 또는 여러 큐비트(리스트)
        """
        self.wires = [wires] if isinstance(wires, int) else wires

    def forward(self, q_device: tq.QuantumDevice, wires=None, params=None, inverse=None):
        """연산자를 양자 상태에 적용
        
        Args:
            q_device: 양자 상태를 저장하는 디바이스
            wires: 연산을 적용할 큐비트(들)
            params: 게이트 파라미터
            inverse: 역연산 여부
        """
        # 파라미터가 주어진 경우 업데이트
        if params is not None:
            self.params = params

        # 파라미터 형태 조정 (배치 처리를 위해)
        if self.params is not None:
            self.params = self.params.unsqueeze(-1) if self.params.dim() == 1 else self.params

        # 와이어(큐비트) 정보가 주어진 경우 업데이트
        if wires is not None:
            wires = [wires] if isinstance(wires, int) else wires
            self.wires = wires

        # 실제 게이트 연산 수행 (파라미터가 없는 경우)
        if self.params is None:
            if self.n_wires is None:
                self.func(q_device, self.wires, inverse=self.inverse)
            else:
                self.func(q_device, self.wires, n_wires=self.n_wires, inverse=self.inverse)
        # 파라미터가 있는 경우의 게이트 연산
        else:
            if self.n_wires is None:
                self.func(q_device, self.wires, params=self.params, inverse=self.inverse)
            else:
                self.func(q_device, self.wires, params=self.params, n_wires=self.n_wires, inverse=self.inverse)