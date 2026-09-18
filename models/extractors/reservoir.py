from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.random import random_circuit


qc = random_circuit(num_qubits=4, depth=5, measure=True, seed=42)
print(qc.draw(output="text"))

sampler = StatevectorSampler(seed=42)
result = sampler.run([qc], shots=1024).result()
print(result[0].data.c.get_counts())
