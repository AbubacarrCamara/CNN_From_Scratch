import pandas as pd
import numpy as np 


class Tensor:
   
    """
        Tensor class handles:

        Data storage (raw numbers + shapes)
        gradient tracking (who needs grads)
        graph construction (reverse-mode AD)
        convenience operations (reshape, intialises)

        wraps a NumPy array, tracks gradient history for automoatic differentiation.

        Args:
        data: The raw n-dimensional array of floats

        requires_grad: If True, computation graph built so .backward()
                    can compute gradients  
    """

    def __init__(self, data: np.ndarray, requires_grad: bool = False):

        self.data = data.astype(float)
        self.shape = data.shape
        self.requires_grad = requires_grad

        # This variable will hold the gradient of the loss with respect to this tensors values
        # It will later become an array the same shape as self.data once backpropogation writes into it 
        self.grad = None

        # Keeps track of which tensors fed into this one 
        # As the forward path happens, each iteration will add its tensors to the set
        # so the graph can be traced backwords
        self._prev = set()

        # Records a short label for the operation that created this tensor 
        # eg (add, matmul, conv2d)
        self._op = None

        # Points to a small function that, when called, uses this tensors
        # .grad values to compute and distribute gradients into each tensor in _prev. 
        self._backward_fn = lambda: None 

        def __repr__(self):
            return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

class Layer:
    """
        This class is an abstract base layer for all layers and will handle 
        forward and back computations 
    """
    def __init__(self):

        self.input = None
        self.output = None
        self.params = []

    def forward(self, x : Tensor) -> Tensor:
        # Acceptance of incoming activation 
        self.input  = x

        # Performs the layers actual math (conv2d, dense, activation)
        # Each subclass must implement _forward() to do its own computation 
        out = self._forward(x)

        # Cashing of output for inspection and debugging 
        self.output = out

        # Gives caller the new activation to pass along the network
        return out


    def backward(self, grad_output : Tensor) -> Tensor:
        # Accepts this layers gradient loss output ∂L/∂ and returns this layers loss
        return self._backward(grad_output)
    
class Conv2D:
    
    """
    The convolutional layer uses 3x3 filters to turn input images into output 
    eg (28 x 28) -> (26 x 26 x 8)

    """

    def __init__(self, in_channel, out_channel, kernal_size, bias = True):
        # Creates weight tensor 
        # Shape (out_channel, in_channel, kernal_h, kernal_w)
        # each weight[k] is the 3d filter applied to all in_channels
        
        self.weight = Tensor.randn(
            out_channel,
            in_channel, 
            kernal_size[0],
            kernal_size[1],
            requires_grad=True)
        
        # Registers it so the optimsers can update it
        self.params.append(self.weight)

        # Initialises the one bias per output channel
        if bias:
            self.bias = Tensor.zeros((out_channel,),requires_grad=True)
            self.params.append(self.bias)

class MaxPooling2D:
    """
    Reduces the size of image to its max kernal value 
    eg [0, 50, 0, 80] maxpooling -> [80]
    """
    def __init_(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding 
        self._mask = None

    def _forward(self, x):
        """
        Performs the max-pooling forward pass
        Pads the input tensor if self.padding > 0
        records maximun value for each window
        caches the (row, col) location of that max for backward
        """
        # Unpack sizes
        kh, kw = self.kernel_size
        sh, sw = self.stride

        # Saves input for backward()
        self.input = x

        # Pad input
        padded = np.pad(
            x.data,
            ((0,0), (0,0),
             (self.padding[0],) * 2,
             (self.padding[1],) * 2),
             mode='constant',
             constant_values=0
        )

        batch, channels, H, W = padded.shape
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1

        # Allocates output tensor and mask array
        output = Tensor.zeros((batch, channels, out_h, out_w), requires_grad=x.requires_grad)
        self._mask = np.zeros_like(padded, dtype=bool)

        # Slide window, builds output and mask 
        for b in range(batch):
            for c in range(channels):
                for i in range(out_w):
                    for j in range(out_w):
                        y0, x0 = i * sh, j * sw
                        window = padded[b, c, y0:y0+kh, x0:x0+kw]
                        m = window.max()
                        output.data[b,c,i,j] = m
                        # Mask all positions equal to max value
                        self._mask[b,c,y0:y0+kh, x0:x0+kw] != (window == m)

        return output

        
def _backward(self, grad_output):
    # Unpacks the sizes
    kh, kw = self.kernal_size
    sh, sw = self.stride

    # Builds padded gradient array
    padded_grad = np.zeros_like(self._mask, dtype=float)
    batch, channels, out_h, out_w = grad_output.data.shape

    # routes gradients through mask
    for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        y0, x0 = i * sh, j * sw
                        grad_val = grad_output.data[b, c, i, j]
                        padded_grad[b, c, y0:y0+kh, x0:x0+kw] += (
                            self._mask[b, c, y0:y0+kh, x0:x0+kw] * grad_val
                        )

    # crop padding
    pad_h, pad_w = self.padding
    cropped_data = padded_grad[:,
                               :,
                               pad_h : pad_h + self.input.data.shape[2],
                               pad_w : pad_w + self.input.data.shape[3]]
    

    return Tensor(cropped_data, requires_grad=self.input.requires_grad)
