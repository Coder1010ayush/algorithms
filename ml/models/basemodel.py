from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def compute_loss(self, y, y_pred):
        pass

    @abstractmethod
    def compute_gradient(self, X, y, y_pred):
        pass

    @abstractmethod
    def update_parameters(self, grad_w, grad_b, lr):
        pass
