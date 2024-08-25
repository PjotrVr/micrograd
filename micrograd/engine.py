import math

class Value:
    def __init__(self, data, _children=None):
        self.data = data
        self.grad = 0
        
        # Interals
        self._prev = set(_children) if _children is not None else set()
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers for now"
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,)) 
        def _backward(): 
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        graph = []
        visited = set()
        def build_dependancy_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_dependancy_graph(child) 
                graph.append(v)

        build_dependancy_graph(self)

        self.grad = 1.0 
        for node in reversed(graph):
            node._backward()

    def __repr__(self):
        return f'Value(data={self.data})'

if __name__ == '__main__':
    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    print(c)
