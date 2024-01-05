import random
from minitorch.engine import Node

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0.0
    
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self,nin,nonlin=True,act=None):
        self.w=[Node(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Node(0.0)
        self.nonlin=nonlin
        self.activation=act() if act is not None else None  

        '''
        If no custom function is defined, then you have by default relu in place
        To define a custom activation function this is how you do it:

        ```
        class sigmoid():
            def __init__(self, name='custom'):
                self.name='custom'
            
            def __call__(self,x):
                # x is a list of ``Node`` objects....

                out=[1/(1+e**-v) for v in x]
                return out
        ```

        ```
        self.activation = sigmoid('sigmoid')
        .
        .
        .
        def __call__(self,x):
            a=sum(..)
            .
            .
            z=self.activation(a)
            return z
        ```
        '''
        
    
    def __call__(self,x):
        a=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        if(not self.nonlin):
            return a
        z=self.activation(a) if self.activation is not None else a.relu()
        return z
    
    def parameters(self):
        self.w+[self.b]
    
    def __repr__(self) -> str:
        return f"Neuron(n_in = {len(self.w)}, activation = {self.activation.name})"
    

class Layer(Module):
    def __init__(self,nin,nout, **kwargs):
        self.neurons=[Neuron(nin,**kwargs) for _ in range(nout)]
    
    def __call__(self,x):
        out=[n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self,nin,nouts):
        sz=[nin,nouts]
        self.layers=[Layer(sz[i],sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"        