import torch

class Module:
    def __init__(self):
        # Dictionaries zur Verwaltung von Parametern und Submodulen
        self._parameters = {}
        self._modules = {}
        self.training = True  # Trainingsmodus standardmäßig aktiviert

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} ist kein gültiges Modul.")
        self._modules[name] = module

    def register_parameter(self, name, param):
        if param is not None and not isinstance(param, torch.Tensor):
            raise TypeError(f"{param} ist kein gültiger Parameter.")
        self._parameters[name] = param

    def parameters(self):
        """Gibt alle registrierten Parameter zurück."""
        for name, param in self._parameters.items():
            if param is not None:
                yield param
        for module in self._modules.values():
            if module is not None:
                yield from module.parameters()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Die forward-Methode muss in der Unterklasse implementiert werden.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)

    def eval(self):
        """Setzt das Modul in den Evaluationsmodus."""
        self.train(False)

    def __repr__(self):
        """Gibt die Struktur des Moduls aus."""
        module_str = f"{self.__class__.__name__}("
        for name, module in self._modules.items():
            module_str += f"\n  ({name}): {module}"
        module_str += "\n)"
        return module_str
    
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Gewicht und Bias als Parameter registrieren
        self.weight = torch.randn(out_features, in_features, requires_grad=True)
        self.bias = torch.randn(out_features, requires_grad=True)
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

    def forward(self, x):
        # Lineare Transformation: y = x * W^T + b
        return x @ self.weight.T + self.bias


class MyModel(Module):
    def __init__(self):
        super().__init__()
        # Zwei lineare Schichten hinzufügen
        self.fc1 = Linear(4, 3)  # Erste Schicht
        self.fc2 = Linear(3, 2)  # Zweite Schicht
        self.add_module('fc1', self.fc1)
        self.add_module('fc2', self.fc2)

    def forwardd(self, x):
        # Daten durch die Schichten schicken
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    # Modell erstellen
model = MyModel()

# Eingabedaten
x = torch.randn(5, 4)  # 5 Beispiele, 4 Features

# Forward-Pass
output = model(x)
print("Output:", output)

# Zugriff auf Parameter
print("\nParameter:")
for name, param in model.parameters():
    print(f"{name}: {param.size()}")

# Modell in den Evaluationsmodus setzen
model.eval()
print("\nModus (Training):", model.training)
