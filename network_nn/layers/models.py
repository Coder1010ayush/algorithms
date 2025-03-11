# ------------------------- *utf-8 encoding* ----------------------------
from initializer_w import Initializer
from typing import Literal
from layers.module import Module, Sequential
from tensor import Tensor
import numpy as np
from autograd.autodiff import relu, sigmoid, tanh, concat as cat


class Linear(Module):

    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        init_type: Literal[
            "uniform",
            "normal",
            "constent",
            "ones",
            "zeros",
            "xaviour_uniform",
            "xaviour_normal",
            "kaining_uniform",
            "kaining_normal",
            "trunc_normal",
            "orthogonal",
        ] = "uniform",
        meta: dict = {"low": 0.0, "high": 1.0},
        bias_option=True,
    ):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.meta = meta

        self.init_type = init_type
        # initializing weight of the model
        self.initializer = Initializer()
        self.shape = (in_feature, out_feature)
        self.__init_w(shape=self.shape, meta=self.meta)
        self.bias_option = bias_option
        if self.bias_option:
            self.register_parameter(name="weight", param=self.weight)
            self.register_parameter(name="bias", param=self.bias)
        else:
            self.register_parameter(name="weight", param=self.weight)

    def __init_w(self, shape, meta):
        self.weight = self.initializer.forward(
            shape=shape,
            init_type=self.init_type,
            retain_grad=True,
            meta=meta,
        )
        self.bias = self.initializer.ones(
            shape=(shape[1],), dtype=float, retain_grad=True
        )

    def forward(self, x: Tensor):
        if self.bias_option:
            o1 = x.matmul(self.weight) + self.bias
        else:
            o1 = x.matmul(self.weight)
        return o1


class Embedding(Module):

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.num_embeddings = vocab_size
        self.dim = dim
        self.weight = Tensor(
            np.random.randn(self.num_embeddings, self.dim),
            requires_grad=True,
            dtype=np.float32,
        )
        self.register_parameter(name="embedding", param=self.weight)

    def forward(self, idx):
        return Tensor(
            data=self.weight[idx],
            requires_grad=True,
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        strg = f"nn.Embedding{self.weight.data.shape}"
        return strg


class RNNCell(Module):
    """
    weight_ih (torch.Tensor) = the learnable input-hidden weights, of shape (hidden_size, input_size)
    weight_hh (torch.Tensor) = the learnable hidden-hidden weights, of shape (hidden_size, hidden_size)
    bias_ih = the learnable input-hidden bias, of shape (hidden_size)
    bias_hh = the learnable hidden-hidden bias, of shape (hidden_size)

    """

    def __init__(self, input_size, hidden_size, bias_option, non_linear_act) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_option = bias_option
        self.non_linearity = non_linear_act
        self.initializer = Initializer()
        self.w_ih = self.initializer.lecun_uniform(
            shape=(self.input_size, self.hidden_size),
            retain_grad=True,
            n_in=self.hidden_size,
        )
        self.w_hh = self.initializer.lecun_uniform(
            shape=(self.hidden_size, self.hidden_size),
            retain_grad=True,
            n_in=self.hidden_size,
        )

        if bias_option:
            self.bias_ih = self.initializer.lecun_uniform(
                shape=(1, self.hidden_size), n_in=self.hidden_size, retain_grad=True
            )
            self.bias_hh = self.initializer.lecun_uniform(
                shape=(1, self.hidden_size), n_in=self.hidden_size, retain_grad=True
            )

            self.register_parameter(name="w_ih", param=self.w_ih)
            self.register_parameter(name="w_hh", param=self.w_hh)
            self.register_parameter(name="b_ih", param=self.bias_ih)
            self.register_parameter(name="b_hh", param=self.bias_hh)

        else:
            self.register_parameter(name="w_ih", param=self.w_ih)
            self.register_parameter(name="w_hh", param=self.w_hh)

    def forward(self, x: Tensor, hx: Tensor):
        if self.bias_option:
            h_new = (
                x.matmul(self.w_ih) + self.bias_ih + hx.matmul(self.w_hh) + self.bias_hh
            )
            if self.non_linearity == "relu":
                h_out = relu(f=h_new)
                return h_out
            else:
                h_out = sigmoid(f=h_new)
                return h_out
        else:
            h_new = self.w_ih.matmul(x) + self.w_hh.matmul(hx)
            if self.non_linearity == "relu":
                h_out = relu(f=h_new)
                return h_out
            else:
                h_out = sigmoid(f=h_new)
                return h_out

    def __repr__(self):
        strg = f"nn.RNNCell{self.input_size,self.hidden_size}"
        return strg


class RNN:
    def __init__(self) -> None:
        pass


class GRUCell(Module):
    """
            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
            h' = (1 - z) \odot n + z \odot h
            \end{array}

    """

    def __init__(self, input_size, hidden_size, bias_option) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_option = bias_option

        # all three gates that used in grucell
        self.reset_gate = Linear(
            in_feature=self.input_size + self.hidden_size,
            out_feature=self.hidden_size,
            bias_option=True,
        )
        self.forget_gate = Linear(
            in_feature=self.input_size + self.hidden_size,
            out_feature=self.hidden_size,
            bias_option=True,
        )
        self.update_gate = Linear(
            in_feature=self.input_size + self.hidden_size,
            out_feature=self.hidden_size,
            bias_option=True,
        )

        self.add_module("reset_gate", self.reset_gate)
        self.add_module("forget_gate", self.forget_gate)
        self.add_module("update_gate", self.update_gate)

    def forward(self, x: Tensor, hx: Tensor):
        ls = []
        ls.append(x)
        ls.append(hx)
        out = cat(ls, axis=1)
        r_out = sigmoid(inp_tensor=self.reset_gate(out))
        u_out = sigmoid(inp_tensor=self.update_gate(out))
        # print("rout is ", r_out)
        # print(" --------------------------------------- ")
        # print("rout shape is ", r_out.shape(), type(r_out))
        # print("hx shape is ", hx.shape(), type(hx))
        intermediate_out = r_out * hx
        f_out = sigmoid(inp_tensor=self.forget_gate(cat([x, intermediate_out], axis=1)))
        h_new = (Tensor(data=1, requires_grad=True, dtype=np.float32) - u_out) * (
            f_out + (u_out * hx)
        )
        return h_new


class LSTMCell(Module):

    def __init__(
        self, input_size: int, hidden_size: int, bias_option: bool = True
    ) -> None:
        self.initializer = Initializer()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_option = bias_option
        """
            there are four gates in the lstm cell each gate have its 
            own weight and bias 
        """
        # forget gate
        if self.bias_option:
            self.w_if = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_hf = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.b_f = self.initializer.lecun_normal(
                shape=(1, self.hidden_size), n_in=self.hidden_size, retain_grad=True
            )

            # input gate
            self.w_ii = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_hi = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.b_i = self.initializer.lecun_normal(
                shape=(1, self.hidden_size), n_in=self.hidden_size, retain_grad=True
            )

            # g gate
            self.w_ig = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_hg = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.b_g = self.initializer.lecun_normal(
                shape=(1, self.hidden_size), n_in=self.hidden_size, retain_grad=True
            )

            # o gate
            self.w_io = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_ho = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.b_o = self.initializer.lecun_normal(
                shape=(1, self.hidden_size), n_in=self.hidden_size, retain_grad=True
            )
            self.register_parameter(name="w_if", param=self.w_if)
            self.register_parameter(name="w_hf", param=self.w_hf)
            self.register_parameter(name="w_ii", param=self.w_ii)
            self.register_parameter(name="w_hi", param=self.w_hi)
            self.register_parameter(name="w_ig", param=self.w_ig)
            self.register_parameter(name="w_hg", param=self.w_hg)
            self.register_parameter(name="w_io", param=self.w_io)
            self.register_parameter(name="w_ho", param=self.w_ho)
            self.register_parameter("b_o", self.b_o)
            self.register_parameter("b_g", self.b_g)
            self.register_parameter("b_i", self.b_i)
            self.register_parameter("b_f", self.b_f)
        else:
            self.w_if = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_hf = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            # input gate
            self.w_ii = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_hi = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            # g gate
            self.w_ig = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_hg = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            # o gate
            self.w_io = self.initializer.lecun_uniform(
                shape=(self.input_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.w_ho = self.initializer.lecun_uniform(
                shape=(self.hidden_size, self.hidden_size),
                n_in=self.hidden_size,
                retain_grad=True,
            )
            self.register_parameter(name="w_if", param=self.w_if)
            self.register_parameter(name="w_hf", param=self.w_hf)
            self.register_parameter(name="w_ii", param=self.w_ii)
            self.register_parameter(name="w_hi", param=self.w_hi)
            self.register_parameter(name="w_ig", param=self.w_ig)
            self.register_parameter(name="w_hg", param=self.w_hg)
            self.register_parameter(name="w_io", param=self.w_io)
            self.register_parameter(name="w_ho", param=self.w_ho)

    def forward(self, x: Tensor, hx: list):
        h_prev = hx[0]
        c_prev = hx[1]
        if self.bias_option:
            i_out = (
                x.matmul(self._parameters["w_ii"])
                + self._parameters["b_i"]
                + h_prev.matmul(self._parameters["w_hi"])
            )
            f_out = (
                x.matmul(self._parameters["w_if"])
                + self._parameters["b_f"]
                + h_prev.matmul(self._parameters["w_hf"])
            )
            g_out = (
                x.matmul(self._parameters["w_ig"])
                + self._parameters["b_g"]
                + h_prev.matmul(self._parameters["w_hg"])
            )
            o_out = (
                x.matmul(self._parameters["w_io"])
                + self._parameters["b_o"]
                + h_prev.matmul(self._parameters["w_ho"])
            )
            out = (f_out * c_prev) + (i_out * g_out)
            h_out = o_out * (tanh(f=out))
            h_t = o_out * tanh(h_out)
            return h_out, h_t
        else:
            i_out = x.matmul(self._parameters["w_ii"]) + h_prev.matmul(
                self._parameters["w_hi"]
            )
            f_out = x.matmul(self._parameters["w_if"]) + h_prev.matmul(
                self._parameters["w_hf"]
            )
            g_out = x.matmul(self._parameters["w_ig"]) + h_prev.matmul(
                self._parameters["w_hg"]
            )
            o_out = x.matmul(self._parameters["w_io"]) + h_prev.matmul(
                self._parameters["w_ho"]
            )
            out = (f_out * c_prev) + (i_out * g_out)
            h_out = o_out * (tanh(f=out))
            h_t = o_out * tanh(h_out)
            return h_out, h_t


class Dropout(Module):
    """
    it helps to less overfit the model

    Args:
        Module (_type_): _description_
        self.dropout = float
        self.mask = np.ndarray
    """

    def __init__(self, dropout: float) -> None:
        self.dropout = dropout
        self.mask = None

    def forward(self, x: Tensor):
        self.mask = Tensor(
            data=(np.random.rand(*x.shape()) > self.dropout) / (1.0 - self.dropout),
            dtype=np.float32,
            retain_grad=True,
        )
        return self.mask * x


class Residual(Module):
    """
        Adding previous output to current output.
    Args:
        Module (_type_): _description_
        fn : Module
    """

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor):
        out = x + self.fn(x)
        return out
