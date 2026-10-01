from sklearn.preprocessing import FunctionTransformer
import pennylane as qml

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit(X):
    qml.AngleEmbedding(X,wires=[0,1])
    return qml.expval(qml.PauliZ(wires=[0]))

ft = FunctionTransformer(circuit)
print(ft.transform([[2,3],[4,5]]))