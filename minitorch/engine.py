import math

class Node:
    '''
    Stores a single scalar and its gradient with respect to output
    It is a node in a computational graph
    '''

    def __init__(self, data, _children=(),_op=''):
        self.data=data

        self.grad=0.0
        self._backward=lambda:None  # function to propagate gradient backwards

        self._prev=set(_children)
        self._op=_op                # op is the operation used to create node from prev nodes

    def __add__(self,other):
        other=other if isinstance(other,Node) else Node(other)
        out=Node(self.data+other.data,(self,other),'+')

        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out
    
    def __mul__(self,other):
        other=other if isinstance(other,Node) else Node(other)
        out=Node(self.data*other.data,(self,other),'*')

        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
    
    def __pow__(self,other:(int,float)):
        assert isinstance(other,(int,float)), "Only Supporting int,float for now"
        out=Node(self.data**other,(self,),f'^{other}')

        def _backward():
            self.grad+=other*out.data*(self.data**-1)*out.grad
        out._backward=_backward
        return out

    def relu(self):
        out=Node(0.0 if self.data<0 else self.data,(self,),'ReLU')

        def _backward():
            self.grad+=(out.data>0)*out.grad
        out._backward=_backward
        return out
    
    def exp(self,a=math.e):
        # compute a^self
        out=Node(a**self.data,(self,),f'{a}^')

        def _backward():
            self.grad+=out.data*math.log(a)*out.grad
        out._backward=_backward
        return out

    # backprop in computation graph
    def backward(self):
        # toposort the computation graph from this node
        topo=[]
        vis=set()
        def toposort(v:Node):
            if v not in vis:
                vis.add(v)
                topo.append(v)
                for child in v._prev:
                    toposort(child)
        toposort(self)

        self.grad=1.0
        for v in topo:
            v._backward()
    
    def __neg__(self):
        return self*-1
    
    def __radd__(self,other):
        return self+other

    def __sub__(self,other):
        return self+(-other)
    
    def __rsub__(self,other):
        return other+(-self)
    
    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self,other):
        return self*(other**-1)
    
    def __rtruediv__(self,other):
        return other*(self**-1)
    
    def __repr__(self) -> str:
        return f"Node(data = {self.data}, grad = {self.grad})"
    


        