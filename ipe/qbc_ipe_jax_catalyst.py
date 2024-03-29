import jax
import jax.numpy as jnp
import pennylane as qml
from catalyst import qjit, measure, cond, for_loop, while_loop


def qbc_ipe_algorithm(x, y, num_t_wires=8, num_shots=None):
    assert len(x) == len(y)
    num_n_wires = int(jnp.ceil(jnp.log2(len(x))))
    num_tot_wires = num_t_wires + num_n_wires + 1
    t_wires = range(0, num_t_wires)
    n_wires = range(num_t_wires, num_t_wires + num_n_wires)
    a_wires = range(num_t_wires + num_n_wires, num_t_wires + num_n_wires + 1)
    tot_wires = range(0, num_tot_wires)
    dev = qml.device("lightning.qubit", wires=tot_wires, shots=num_shots)
    tmp = jnp.zeros((2**num_n_wires, 2**num_n_wires - 1))
    A_x = jnp.concatenate(
        (jnp.pad(x, pad_width=(0, 2**num_n_wires - len(x))).reshape(-1, 1), tmp),
        axis=1,
    )
    Q_x, _ = jnp.linalg.qr(A_x, mode="full")
    for i in range(2**num_n_wires):
        Q_x.at[:, i].set(Q_x[:, i] / jnp.linalg.norm(Q_x[:, i]))
    u_x = Q_x.T
    u_x *= jnp.sign(u_x[0, 0]) * jnp.sign(x[0])

    A_y = jnp.concatenate(
        (jnp.pad(y, pad_width=(0, 2**num_n_wires - len(y))).reshape(-1, 1), tmp),
        axis=1,
    )
    Q_y, _ = jnp.linalg.qr(A_y, mode="full")
    for i in range(2**num_n_wires):
        Q_y.at[:, i].set(Q_y[:, i] / jnp.linalg.norm(Q_y[:, i]))
    u_y = Q_y.T
    u_y *= jnp.sign(u_y[0, 0]) * jnp.sign(y[0])

    # QBC circuit
    @qjit
    @qml.qnode(dev, interface="jax")
    def qbc_ipe():
        # Oracle A encoding x data
        @jax.jit
        def U_x():
            qml.QubitUnitary(u_x, wires=n_wires)

        # Oracle B encoding y data
        @jax.jit
        def U_y():
            qml.QubitUnitary(u_y, wires=n_wires)

        def A_operator():
            qml.Hadamard(wires=a_wires)
            qml.ctrl(U_x, control=a_wires, control_values=[0])()
            qml.PauliX(wires=a_wires)
            qml.ctrl(U_y, control=a_wires, control_values=[0])()
            qml.PauliX(wires=a_wires)
            qml.Hadamard(wires=a_wires)

        def ctrl_grover_operator(control):
            """Controlled Grover operator using only gates supported by lightning.qubit"""
            # Reflection about "good" state
            qml.CNOT(wires=[control, a_wires[0]])
            qml.PauliZ(wires=a_wires)
            qml.CNOT(wires=[control, a_wires[0]])

            A_operator()

            # Reflection about |0> state
            for i in n_wires:
                qml.PauliX(wires=i)
            qml.PauliX(wires=a_wires)
            qml.CNOT(wires=[control, a_wires[0]])
            qml.Hadamard(wires=a_wires)
            qml.ctrl(
                qml.PauliX,
                control=n_wires,
                control_values=jnp.ones(num_n_wires),
            )(wires=a_wires)
            qml.Hadamard(wires=a_wires)
            qml.CNOT(wires=[control, a_wires[0]])
            for i in n_wires:
                qml.PauliX(wires=i)
            qml.PauliX(wires=a_wires)

            qml.adjoint(A_operator)()

        A_operator()

        for t in t_wires:
            qml.Hadamard(wires=t)

        for idx, t in enumerate(t_wires):
            for i in range(2 ** (num_t_wires - idx - 1)):
                ctrl_grover_operator(control=t)

        qml.adjoint(qml.QFT)(wires=t_wires)

        return qml.probs(wires=t_wires)

    probs = qbc_ipe()
    j = jnp.argmax(probs)

    print("theta =", 2 * j / (2**num_t_wires))
    rho = (
        -(1.0 - 2.0 * jnp.sin(jnp.pi * j / (2**num_t_wires)) ** 2)
        * jnp.linalg.norm(x)
        * jnp.linalg.norm(y)
    )

    return rho


x = jnp.array([0.5, -1.0])
y = jnp.array([1.0, -1.75])

x /= jnp.linalg.norm(x)
y /= jnp.linalg.norm(y)
result = qbc_ipe_algorithm(x, y, num_t_wires=6, num_shots=1)
print("x =", x, "y =", y, "result =", result, "result_exact =", jnp.inner(x, y))
