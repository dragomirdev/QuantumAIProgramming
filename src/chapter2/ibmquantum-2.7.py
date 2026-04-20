from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(token="<<YOUR TOKEN>>",
                                  instance="<<YOUR INSTANCE>>")

service = QiskitRuntimeService()
backend = service.least_busy(simulator=False)

print("Using backend:", backend)