import numpy.testing as npt
from typing import Tuple, Union
import torch
import numpy as np


class BroadcastFunction:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def forward(self, array, target_shape):
        """
        Broadcasts an array to a target shape if possible.

        Args:
            array: numpy array to broadcast
            target_shape: desired output shape

        Returns:
            Broadcasted array if possible, else raises ValueError
        """
        self.input_shape = array.shape
        self.output_shape = target_shape

        # Handle the case where we need to add dimensions in the middle
        current_shape = list(array.shape)
        target = list(target_shape)

        # First add leading dimensions
        while len(current_shape) < len(target):
            current_shape.insert(0, 1)
        array = array.reshape(current_shape)

        # For each dimension, broadcast as needed
        broadcasted = np.array(array)  # Make a copy to avoid modifying original
        for axis in range(len(target)):
            if broadcasted.shape[axis] == 1 and target[axis] > 1:
                broadcasted = np.repeat(broadcasted, target[axis], axis=axis)

        return broadcasted

    def backward(self, grad):
        """
        Computes gradient with respect to input array.

        Args:
            grad: gradient of the same shape as broadcasted output

        Returns:
            Gradient with respect to input array
        """
        if grad.shape != self.output_shape:
            raise ValueError(f"Gradient shape {grad.shape} doesn't match broadcast shape {self.output_shape}")

        # Pad input shape with ones if needed
        current_shape = list(self.input_shape)
        while len(current_shape) < len(self.output_shape):
            current_shape.insert(0, 1)

        # Find axes where broadcasting occurred
        reduction_axes = []
        for i, (a, b) in enumerate(zip(current_shape, self.output_shape)):
            if a != b:
                if a != 1:
                    raise ValueError(f"Invalid broadcast: dim {i} has sizes {a} and {b}")
                reduction_axes.append(i)

        # Sum over broadcasted axes
        if reduction_axes:
            grad = np.sum(grad, axis=tuple(reduction_axes), keepdims=True)

        # Reshape back to input shape
        return grad.reshape(self.input_shape)


def test_broadcast_function():
    bf = BroadcastFunction()

    print("Running test cases...")

    # Test case 1: Broadcasting scalar to array
    x = np.array([1.0])
    target_shape = (2, 3)
    forward = bf.forward(x, target_shape)
    assert forward.shape == target_shape
    grad = np.ones(target_shape)
    backward = bf.backward(grad)
    assert backward.shape == x.shape
    print("Test 1 passed: Scalar to array")

    # Test case 2: Broadcasting 1D to 2D
    x = np.array([1., 2., 3.])
    target_shape = (2, 3)
    forward = bf.forward(x, target_shape)
    assert forward.shape == target_shape
    grad = np.ones(target_shape)
    backward = bf.backward(grad)
    assert backward.shape == x.shape
    print("Test 2 passed: 1D to 2D")

    # Test case 3: Broadcasting column vector
    x = np.array([[1.], [2.]])  # Shape (2, 1)
    target_shape = (2, 3)
    forward = bf.forward(x, target_shape)
    assert forward.shape == target_shape
    grad = np.ones(target_shape)
    backward = bf.backward(grad)
    assert backward.shape == x.shape
    print("Test 3 passed: Column vector")

    # Test case 4: Complex broadcasting
    x = np.array([[1., 2.]])  # Shape (1, 2)
    target_shape = (3, 2, 4)
    forward = bf.forward(x, target_shape)
    assert forward.shape == target_shape
    grad = np.ones(target_shape)
    backward = bf.backward(grad)
    assert backward.shape == x.shape
    print("Test 4 passed: Complex broadcasting")

    print("All tests passed!")


class BroadcastValidator:
    @staticmethod
    def numpy_forward_backward(
        array: np.ndarray,
        target_shape: Tuple[int, ...],
        grad: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward and backward pass using our custom numpy implementation
        """
        bf = BroadcastFunction()
        forward_output = bf.forward(array, target_shape)
        backward_output = bf.backward(grad)
        return forward_output, backward_output

    @staticmethod
    def torch_forward_backward(
        array: Union[np.ndarray, torch.Tensor],
        target_shape: Tuple[int, ...],
        grad: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forward and backward pass using PyTorch autograd
        """
        # Convert inputs to torch tensors if they're numpy arrays
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array.copy()).float()
        if isinstance(grad, np.ndarray):
            grad = torch.from_numpy(grad.copy()).float()

        # Create a leaf tensor that requires gradients
        array = array.clone().detach().requires_grad_(True)

        # Add dimensions to match target shape
        tensor = array
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(0)

        # Forward pass using expand
        # Note: we create a new tensor here to ensure proper gradient flow
        forward_output = tensor.expand(target_shape).clone()

        # Backward pass
        forward_output.backward(grad)
        backward_output = array.grad  # Now array.grad will exist

        return forward_output, backward_output

    @staticmethod
    def validate_broadcast(
        input_array: np.ndarray,
        target_shape: Tuple[int, ...],
        grad_array: np.ndarray,
        rtol: float = 1e-5,
        atol: float = 1e-5
    ) -> bool:
        """
        Validate broadcasting by comparing numpy and torch implementations
        """
        print(f"\nValidating broadcast from shape {input_array.shape} to {target_shape}")

        try:
            # Get results from both implementations
            np_forward, np_backward = BroadcastValidator.numpy_forward_backward(
                input_array.copy(), target_shape, grad_array.copy()
            )
            torch_forward, torch_backward = BroadcastValidator.torch_forward_backward(
                input_array.copy(), target_shape, grad_array.copy()
            )

            # Convert torch outputs to numpy for comparison
            torch_forward = torch_forward.detach().numpy()
            torch_backward = torch_backward.detach().numpy()

            # Compare forward pass
            npt.assert_allclose(np_forward, torch_forward, rtol=rtol, atol=atol)
            print("✓ Forward pass matches")

            # Compare backward pass
            npt.assert_allclose(np_backward, torch_backward, rtol=rtol, atol=atol)
            print("✓ Backward pass matches")

            return True

        except Exception as e:
            print(f"✗ Validation failed: {str(e)}")
            try:
                if 'np_backward' in locals() and 'torch_backward' in locals():
                    print("\nNumPy output shape:", np_backward.shape)
                    print("NumPy output:\n", np_backward)
                    print("\nPyTorch output shape:", torch_backward.shape)
                    print("PyTorch output:\n", torch_backward)
            except:
                pass
            return False


def run_validation_tests():
    """
    Run a comprehensive set of validation tests
    """
    validator = BroadcastValidator()

    test_cases = [
        # Test 1: Scalar to 2D
        {
            'input': np.array([1.0]),
            'target_shape': (2, 3),
            'grad': np.ones((2, 3))
        },

        # Test 2: 1D to 2D
        {
            'input': np.array([1., 2., 3.]),
            'target_shape': (2, 3),
            'grad': np.ones((2, 3))
        },

        # Test 3: Column vector to 2D
        {
            'input': np.array([[1.], [2.]]),
            'target_shape': (2, 3),
            'grad': np.ones((2, 3))
        },

        # Test 4: Complex broadcasting
        {
            'input': np.array([[1., 2.]]),
            'target_shape': (3, 2, 4),
            'grad': np.ones((3, 2, 4))
        },

        # Test 5: Broadcasting with non-uniform gradients
        {
            'input': np.array([1., 2., 3.]),
            'target_shape': (2, 3),
            'grad': np.array([[1., 2., 3.],
                              [4., 5., 6.]])
        },

        # Test 6: High dimensional broadcasting
        {
            'input': np.array([[[1.]]]),
            'target_shape': (2, 3, 4),
            'grad': np.ones((2, 3, 4))
        }
    ]

    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}")
        print("-" * 40)
        success = validator.validate_broadcast(
            test_case['input'].copy(),
            test_case['target_shape'],
            test_case['grad'].copy()
        )
        all_passed &= success

    if all_passed:
        print("\n✓ All validation tests passed!")
    else:
        print("\n✗ Some validation tests failed!")


def demonstrate_usage():
    """
    Demonstrate practical usage with a few examples
    """
    print("\nDemonstrating practical usage:")
    print("-" * 40)

    # Example 1: Broadcasting in neural network bias addition
    input_bias = np.array([1., 2., 3.])
    feature_shape = (32, 3)  # 32 samples, 3 features
    grad = np.random.randn(*feature_shape)

    print("\nValidating bias addition broadcasting:")
    BroadcastValidator.validate_broadcast(input_bias, feature_shape, grad)

    # Example 2: Broadcasting in batch normalization
    scale = np.array([1., 2., 3., 4.]).reshape(1, 4, 1, 1)
    feature_map_shape = (32, 4, 28, 28)  # batch, channels, height, width
    grad = np.random.randn(*feature_map_shape)

    print("\nValidating batch normalization broadcasting:")
    BroadcastValidator.validate_broadcast(scale, feature_map_shape, grad)


if __name__ == "__main__":
    # Run all validation tests
    run_validation_tests()

    # Show practical usage examples
    demonstrate_usage()
    """  
    # tpns = ['[REP]','[DEL]', '[PRO]' , '[INS]' , 'u', 'bʰ', 'ɳ', 'sh', 'ŋ', 'ɡʰ', 'ɔ̃', 'ʈ', 'i', 'h', 'g', 'ɟ', 'd', 'ʌ', 'ɾ', 'ẽ', 'o', 'c', 's', 'ʋ', 'ɖ', 'r', 'ɛ', 'tʰ', 'pʰ', 'cʰ', 'n', 'õ', 'ʊ', 'ʃ', 'a', 'ʈʰ', 'b', 't', 'ə', 'ɡ', 'dʰ', 'p', 'k', 'l', 'ɟʰ', 'kʰ', 'ĩ', 'ɖʰ', 'ã', 'ɔ', 'e', 'ʂ', 'f', 'm', 'ɪ', 'ɲ', 'j']
# rare_ipa_phonemes = ['ʡ', 'ʢ', 'ʞ', 'ʘ', 'ǂ', 'ʍ̥']
# unknown_tokens = tpns + rare_ipa_phonemes[0:6]
unknown_tokens = ['[REP]','[DEL]', '[PRO]' , '[INS]', 'z', 'ɔ', 'ɪ', 'ʋ', 'ch', 'ɣ', 'th', 'eʰ', 'ng', 
 'd', 'oʰ', 'ih', 'ʃ', 'kʰ', 's', 'b', 'bʱ', 'cʰ', 'ʂ', 'n', 'ɾʰ', 'ɡ', 
 'x', 'dʰ', 'c', 'ɦ', 'ĩ', 'jh', 'e', 'ɾ', 
 'j', 'ɡʰ', 'k', 'uʰ', 'ɖʰ', 't', 'ʊ', 'tʰ', 'pʰ', 'dʒʱ', 'o', 'r', 
 'a', 'u', 'χ', 'ɪʰ', 'bʰ', 'q', 'dʒ', 'l', 'p', 'ʰ', 'm', 'i', 'ə', 
 'tʃ', 'f', 'h', 'ʈ', 'uh', '̃', 'ɡʱ', 'ɟ', 'ʌ', 'w' , 'ʈʰ' , 'ae']

    
    """
