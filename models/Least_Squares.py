# %% Cell to run all models

# We want to look at direct minimisation as a function of the noise levels and see how it changes for different models
from shapely.ops import cascaded_union
from shapely.geometry import Point
import matplotlib.patches as ptc
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import laplace
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# Define models

# Schnakenberg


def step_forward_schnakenberg_nd3(y, t, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = laplacianu + c[0] - u + c[1] * u**2 * v
    dvdt = c[2] * laplacianv + c[3] - c[1] * u**2 * v

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


def step_forward_schnakenberg_nd3_ivp(t, y, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = laplacianu + c[0] - u + c[1] * u**2 * v
    dvdt = c[2] * laplacianv + c[3] - c[1] * u**2 * v

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


# FitzHugh-Nagumo
def step_forward_FN_nd(y, t, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = laplacianu + v - c[0] * u
    dvdt = c[1] * laplacianv + c[2] * v - c[3] * u - v**3

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


def step_forward_FN_nd_ivp(t, y, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = laplacianu + v - c[0] * u
    dvdt = c[1] * laplacianv + c[2] * v - c[3] * u - v**3

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


# # Brusselator
def step_forward_B_nd(y, t, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = c[0] * laplacianu + 1 - u + c[1] * u**2 * v
    dvdt = laplacianv + c[2] * u - c[1] * u**2 * v

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


def step_forward_B_nd_ivp(t, y, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = c[0] * laplacianu + 1 - u + c[1] * u**2 * v
    dvdt = laplacianv + c[2] * u - c[1] * u**2 * v

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


# Brusselator 2
def step_forward_B_nd_v2(y, t, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = laplacianu + 1 - c[0] * u + c[1] * u**2 * v
    dvdt = c[2] * laplacianv + u - c[1] * u**2 * v

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


def step_forward_B_nd_v2_ivp(t, y, c, dx):
    # The vectors u and v are interleaved in y.  We define
    # views of u and v by slicing y.
    # And we can recover the matrices by doing a reshape
    n = int(np.sqrt(len(y) / 2))
    u = np.reshape(y[::2], (n, n))
    v = np.reshape(y[1::2], (n, n))

    # dydt is the return value of this function.
    dydt = np.empty_like(y)
    # Just like u and v are views of the interleaved vectors
    # in y, dudt and dvdt are views of the interleaved output
    # vectors in dydt.
    laplacianu = laplace(u, mode="nearest") / dx**2
    laplacianv = laplace(v, mode="nearest") / dx**2
    # Compute du/dt and dv/dt.  The end points and the interior points
    # are handled separately.
    dudt = laplacianu + 1 - c[0] * u + c[1] * u**2 * v
    dvdt = c[2] * laplacianv + u - c[1] * u**2 * v

    dydt[::2] = np.reshape(dudt, n**2)
    dydt[1::2] = np.reshape(dvdt, n**2)
    return dydt


# Define functions that compute the patterns given parameters


def gen_pattern_sch_ivp(params, perturbation1, perturbation2, n=50, t=80000):
    c_S = [1, 40, 1, 0.1, 0.9, 1]  # S
    u_tf = v_tf = c_S[4] / c_S[2]
    x_tf = np.sqrt(c_S[2] / c_S[0])
    dx_sch = 1 * x_tf
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u_tf * (u0 + perturbation1)
    y0[1::2] = v_tf * (v0 + perturbation2)
    tlen = int(t)
    t = np.linspace(0, tlen, tlen)
    solb = solve_ivp(
        step_forward_schnakenberg_nd3_ivp,
        [t[0], t[-1]],
        y0,
        args=(params, dx_sch),
        rtol=1e-8,
        atol=1e-8,
    ).y[:, -1]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    return u_tp, v_tp


def gen_pattern_sch(params, perturbation1, perturbation2, n=50, t=80000):
    c_S = [1, 40, 1, 0.1, 0.9, 1]  # S
    u_tf = v_tf = c_S[4] / c_S[2]
    x_tf = np.sqrt(c_S[2] / c_S[0])
    dx_sch = 1 * x_tf
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u_tf * (u0 + perturbation1)
    y0[1::2] = v_tf * (v0 + perturbation2)
    tlen = int(t)
    t = np.linspace(0, tlen, tlen)
    solb = odeint(
        step_forward_schnakenberg_nd3, y0, t, args=(params, dx_sch), ml=2 * n, mu=2 * n
    )[-1]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    # u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    # v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    return u_tp, v_tp
    # return solb


def gen_pattern_fn_ivp(params, perturbation1, perturbation2, n=50, t=400000):
    c_FN = [0.05, 0.00028, 1, 10, 1]  # FN du dv alpha eps mu
    u_tf = v_tf = 1 / np.sqrt(c_FN[3])
    x_tf = 1 / np.sqrt(c_FN[0] / c_FN[3])
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = u_tf*np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = v_tf*np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    # y0[::2] = c_S[4]*c_S[5]/c_S[0]**2*(u0+perturbation1)
    # y0[1::2] = c_S[0]/c_S[4]*(v0+perturbation2)
    y0[::2] = u0 + u_tf * perturbation1
    y0[1::2] = v0 + v_tf * perturbation2
    tlen = t
    t = np.linspace(0, tlen, tlen)
    dx_fn = 1 / n * x_tf
    solb = solve_ivp(
        step_forward_FN_nd_ivp,
        [t[0], t[-1]],
        y0,
        args=(params, dx_fn),
        rtol=1e-8,
        atol=1e-8,
    ).y[:, -1]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    return u_tp, v_tp


def gen_pattern_fn(params, perturbation1, perturbation2, n=50, t=400000):
    c_FN = [0.05, 0.00028, 1, 10, 1]  # FN du dv alpha eps mu
    u_tf = v_tf = 1 / np.sqrt(c_FN[3])
    x_tf = 1 / np.sqrt(c_FN[0] / c_FN[3])
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = u_tf*np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = v_tf*np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    # y0[::2] = c_S[4]*c_S[5]/c_S[0]**2*(u0+perturbation1)
    # y0[1::2] = c_S[0]/c_S[4]*(v0+perturbation2)
    y0[::2] = u0 + u_tf * perturbation1
    y0[1::2] = v0 + v_tf * perturbation2
    tlen = t
    t = np.linspace(0, tlen, tlen)
    dx_fn = 1 / n * x_tf
    solb = odeint(step_forward_FN_nd, y0, t, args=(params, dx_fn), ml=2 * n, mu=2 * n)[
        -1
    ]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    return u_tp, v_tp


def gen_pattern_b_ivp(params, perturbation1, perturbation2, n=50):
    c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
    u_tf = v_tf = (c_B[3] - 1) / c_B[2]
    x_tf = 1 / np.sqrt(c_B[1] / (c_B[3] + 1))
    t_tf = c_B[3] + 1
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = u_tf*np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = v_tf*np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u0 + u_tf * perturbation1
    y0[1::2] = v0 + v_tf * perturbation2
    tlen = int(400000)
    t = np.linspace(0, tlen, tlen)
    dx_b = 1 / n * x_tf
    solb = solve_ivp(
        step_forward_B_nd_ivp,
        [t[0], t[-1]],
        y0,
        args=(params, dx_b),
        rtol=1e-8,
        atol=1e-8,
    ).y[:, -1]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    return u_tp, v_tp


def gen_pattern_b(params, perturbation1, perturbation2, n=50):
    c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
    u_tf = v_tf = (c_B[3] - 1) / c_B[2]
    x_tf = 1 / np.sqrt(c_B[1] / (c_B[3] + 1))
    t_tf = c_B[3] + 1
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = u_tf*np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = v_tf*np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u0 + u_tf * perturbation1
    y0[1::2] = v0 + v_tf * perturbation2
    tlen = int(400000)
    t = np.linspace(0, tlen, tlen)
    dx_b = 1 / n * x_tf
    solb = odeint(step_forward_B_nd, y0, t, args=(params, dx_b), ml=2 * n, mu=2 * n)[-1]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    return u_tp, v_tp


def gen_pattern_b_v2_ivp(params, perturbation1, perturbation2, n=50):
    c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
    u_tf = v_tf = (c_B[3]) / c_B[2]
    x_tf = 1 / np.sqrt(c_B[0] / (c_B[3]))
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = u_tf*np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = v_tf*np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u0 + u_tf * perturbation1
    y0[1::2] = v0 + v_tf * perturbation2
    tlen = int(400000)
    t = np.linspace(0, tlen, tlen)
    dx_b_v2 = 1 / n * x_tf
    solb = solve_ivp(
        step_forward_B_nd_v2_ivp,
        [t[0], t[-1]],
        y0,
        args=(params, dx_b_v2),
        rtol=1e-8,
        atol=1e-8,
    ).y[:, -1]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    # We do a copy since it seems that if we don't, we keep references to
    # solb which keeps a lot of information on the time, integration etc
    # which took a lot of memory and ended up causing a crash
    return u_tp, v_tp


def gen_pattern_b_v2(params, perturbation1, perturbation2, n=50):
    c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
    u_tf = v_tf = (c_B[3]) / c_B[2]
    x_tf = 1 / np.sqrt(c_B[0] / (c_B[3]))
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    # perturbation1 = u_tf*np.random.normal(0, 0.0001, (n**2))
    # perturbation2 = v_tf*np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u0 + u_tf * perturbation1
    y0[1::2] = v0 + v_tf * perturbation2
    tlen = int(400000)
    t = np.linspace(0, tlen, tlen)
    dx_b_v2 = 1 / n * x_tf
    solb = odeint(
        step_forward_B_nd_v2, y0, t, args=(params, dx_b_v2), ml=2 * n, mu=2 * n
    )[-1]
    u_tp = np.copy(np.reshape(solb[::2], (n, n)))
    v_tp = np.copy(np.reshape(solb[1::2], (n, n)))
    # We do a copy since it seems that if we don't, we keep references to
    # solb which keeps a lot of information on the time, integration etc
    # which took a lot of memory and ended up causing a crash
    return u_tp, v_tp


# Get patterns
n = 50
# Initialise perturbations
perturbation_1 = np.random.normal(0, 0.0001, (n**2))
perturbation_2 = np.random.normal(0, 0.0001, (n**2))
perturbation_1, perturbation_2 = np.load("Perturbation_arrays.npy")

# np.save('Perturbation_arrays',np.vstack((perturbation_1,perturbation_2)))
# Saving perturbation arrays so that we can recover everything at any point and use the same for all parts.
c_S = [1, 40, 1, 0.1, 0.9, 1]  # S
c_original_sch = [
    (c_S[3] * c_S[4]) / c_S[2] ** 2,
    c_S[5] * c_S[2] / (c_S[4] ** 2),
    c_S[1] / c_S[0],
    1,
]  # S nd
u_tf = v_tf = c_S[4] / c_S[2]
x_tf = np.sqrt(c_S[2] / c_S[0])
dx_sch = 1 * x_tf

c_FN = [0.05, 0.00028, 1, 10, 1]  # FN du dv alpha eps mu
c_original_fn = [c_FN[2], c_FN[1] / c_FN[0], c_FN[4] / c_FN[3], 1 / c_FN[3]]
u_tf = v_tf = 1 / np.sqrt(c_FN[3])
x_tf = 1 / np.sqrt(c_FN[0] / c_FN[3])
dx_fn = 1 / n * x_tf


c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
c_original_b = [c_B[0] / c_B[1], c_B[2] ** 2 / (c_B[3] + 1) ** 3, c_B[3] / (c_B[3] + 1)]
x_tf = 1 / np.sqrt(c_B[1] / (c_B[3] + 1))
dx_b = 1 / n * x_tf
c_original_b_v2 = [
    (c_B[3] + 1) / (c_B[3]),
    c_B[2] ** 2 / (c_B[3]) ** 3,
    c_B[1] / c_B[0],
]
u_tf = v_tf = (c_B[3]) / c_B[2]
x_tf = 1 / np.sqrt(c_B[0] / (c_B[3]))
dx_b_v2 = 1 / n * x_tf


u_tp_sch, v_tp_sch = gen_pattern_sch_ivp(c_original_sch, perturbation_1, perturbation_2)
u_tp_fn, v_tp_fn = gen_pattern_fn_ivp(c_original_fn, perturbation_1, perturbation_2)
u_tp_b, v_tp_b = gen_pattern_b_ivp(c_original_b, perturbation_1, perturbation_2)
u_tp_b_v2, v_tp_b_v2 = gen_pattern_b_v2_ivp(
    c_original_b_v2, perturbation_1, perturbation_2
)
plt.imshow(u_tp_sch)

u_tp_sch, v_tp_sch = gen_pattern_sch(c_original_sch, perturbation_1, perturbation_2)
u_tp_fn, v_tp_fn = gen_pattern_fn(c_original_fn, perturbation_1, perturbation_2)
u_tp_b, v_tp_b = gen_pattern_b(c_original_b, perturbation_1, perturbation_2)
u_tp_b_v2, v_tp_b_v2 = gen_pattern_b_v2(c_original_b_v2, perturbation_1, perturbation_2)

plt.imshow(u_tp_sch)
plt.imshow(u_tp_fn)
plt.imshow(u_tp_b)
plt.imshow(u_tp_b_v2)
# Get patterns with n= 100
n = 100
# Initialise perturbations
perturbation_1 = np.random.normal(0, 0.0001, (n**2))
perturbation_2 = np.random.normal(0, 0.0001, (n**2))
# perturbation_1, perturbation_2 = np.load('Perturbation_arrays.npy')

# np.save('Perturbation_arrays',np.vstack((perturbation_1,perturbation_2)))
# Saving perturbation arrays so that we can recover everything at any point and use the same for all parts.
c_S = [1, 40, 1, 0.1, 0.9, 1]  # S
c_original_sch = [
    (c_S[3] * c_S[4]) / c_S[2] ** 2,
    c_S[5] * c_S[2] / (c_S[4] ** 2),
    c_S[1] / c_S[0],
    1,
]  # S nd
u_tf = v_tf = c_S[4] / c_S[2]
x_tf = np.sqrt(c_S[2] / c_S[0])
dx_sch = 1 * x_tf

c_FN = [0.05, 0.00028, 1, 10, 1]  # FN du dv alpha eps mu
c_original_fn = [c_FN[2], c_FN[1] / c_FN[0], c_FN[4] / c_FN[3], 1 / c_FN[3]]
u_tf = v_tf = 1 / np.sqrt(c_FN[3])
x_tf = 1 / np.sqrt(c_FN[0] / c_FN[3])
dx_fn = 1 / n * x_tf


c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
c_original_b = [c_B[0] / c_B[1], c_B[2] ** 2 / (c_B[3] + 1) ** 3, c_B[3] / (c_B[3] + 1)]
u_tf = v_tf = (c_B[3] - 1) / c_B[2]
x_tf = 1 / np.sqrt(c_B[1] / (c_B[3] + 1))
dx_b = 1 / n * x_tf

c_original_b_v2 = [
    (c_B[3] + 1) / (c_B[3]),
    c_B[2] ** 2 / (c_B[3]) ** 3,
    c_B[1] / c_B[0],
]
u_tf = v_tf = (c_B[3]) / c_B[2]
x_tf = 1 / np.sqrt(c_B[0] / (c_B[3]))
dx_b_v2 = 1 / n * x_tf


solb = gen_pattern_sch(c_original_sch, perturbation_1, perturbation_2, n, 500)
u_tp_sch = np.copy(np.reshape(solb[:, -1][::2], (n, n)))
np.mean(np.reshape(solb[:, -1][::2], (n, n)) - np.reshape(solb[:, -2][::2], (n, n)))
v_tp_sch = np.copy(np.reshape(solb[:, -1][1::2], (n, n)))
plt.imshow(np.reshape(solb[:, -1][::2], (n, n)) - np.reshape(solb[:, -2][::2], (n, n)))
plt.colorbar()
u_tp_fn, v_tp_fn = gen_pattern_fn(c_original_fn, perturbation_1, perturbation_2, n, 400)
plt.imshow(u_tp_sch)
u_tp_b, v_tp_b = gen_pattern_b(c_original_b, perturbation_1, perturbation_2)
u_tp_b_v2, v_tp_b_v2 = gen_pattern_b_v2(c_original_b_v2, perturbation_1, perturbation_2)

# Define minimisations


def optimiser_sch(u_tp, v_tp, dx):
    # crop the laplacian because of BC
    laplacian_u = laplace(u_tp, mode="nearest").flatten() / dx**2
    laplacian_v = laplace(v_tp, mode="nearest").flatten() / dx**2
    u = u_tp.flatten()
    v = v_tp.flatten()
    X1 = np.vstack((np.ones((n**2)), u**2 * v, np.zeros((2, n**2)))).T
    X2 = np.vstack([np.zeros(n**2), -(u**2) * v, laplacian_v, np.ones(n**2)]).T
    X = np.vstack((X1, X2))
    Y = np.hstack((u - laplacian_u, np.zeros(n**2)))
    d1, d2, d3, d4 = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    return [d1, d2, d3, d4]


def optimiser_fn(u_tp, v_tp, dx):
    # crop the laplacian because of BC
    laplacian_u = laplace(u_tp, mode="nearest").flatten() / dx**2
    laplacian_v = laplace(v_tp, mode="nearest").flatten() / dx**2
    u = u_tp.flatten()
    v = v_tp.flatten()
    X1 = np.vstack((-u, np.zeros((3, n**2)))).T
    X2 = np.vstack([np.zeros(n**2), laplacian_v, v, -u]).T
    X = np.vstack((X1, X2))
    Y = np.hstack((-v - laplacian_u, v**3))
    d1, d2, d3, d4 = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))

    return [d1, d2, d3, d4]


def optimiser_b(u_tp, v_tp, dx):
    # crop the laplacian because of BC
    laplacian_u = laplace(u_tp, mode="nearest").flatten() / dx**2
    laplacian_v = laplace(v_tp, mode="nearest").flatten() / dx**2
    u = u_tp.flatten()
    v = v_tp.flatten()
    X1 = np.vstack((laplacian_u, u**2 * v, np.zeros((n**2)))).T
    X2 = np.vstack([np.zeros(n**2), -(u**2) * v, u]).T
    X = np.vstack((X1, X2))
    Y = np.hstack((u - 1, -laplacian_v))
    d1, d2, d3 = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    return [d1, d2, d3]


def optimiser_b_v2(u_tp, v_tp, dx):
    # crop the laplacian because of BC
    laplacian_u = laplace(u_tp, mode="nearest").flatten() / dx**2
    laplacian_v = laplace(v_tp, mode="nearest").flatten() / dx**2
    u = u_tp.flatten()
    v = v_tp.flatten()
    X1 = np.vstack((-u, u**2 * v, np.zeros((n**2)))).T
    X2 = np.vstack([np.zeros(n**2), -(u**2) * v, laplacian_v]).T
    X = np.vstack((X1, X2))
    Y = np.hstack((-laplacian_u - 1, -u))
    d1, d2, d3 = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    return [d1, d2, d3]


# %%


# Define noise level
n_noise = 500
noise_array = np.geomspace(1e-8, 0.4, n_noise)
rel_error_mat_sch = np.zeros(n_noise)
rel_error_mat_fn = np.zeros(n_noise)
rel_error_mat_b = np.zeros(n_noise)
rel_error_mat_mean_sch = np.zeros(n_noise)
rel_error_mat_mean_fn = np.zeros(n_noise)
rel_error_mat_mean_b = np.zeros(n_noise)
MSE_u_mat_sch = np.zeros(n_noise)
MSE_v_mat_sch = np.zeros(n_noise)
MSE_u_mat_fn = np.zeros(n_noise)
MSE_v_mat_fn = np.zeros(n_noise)
MSE_u_mat_b = np.zeros(n_noise)
MSE_v_mat_b = np.zeros(n_noise)
params_sch = []
params_fn = []
params_b = []
for i, noise_lev in enumerate(noise_array):
    # Add noise to patterns

    # Schnakenberg
    noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
    noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
    noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
    noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
    u_tp_noise_sch = u_tp_sch + noise_u_sch
    v_tp_noise_sch = v_tp_sch + noise_v_sch

    # FitzHugh-Nagumo
    noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
    noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
    noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
    noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
    u_tp_noise_fn = u_tp_fn + noise_u_fn
    v_tp_noise_fn = v_tp_fn + noise_v_fn

    # Brusselator
    # noise_spread_u_b = noise_lev/100*(np.max(u_tp_b)-np.min(u_tp_b))
    # # # noise_spread_v_b = noise_lev/100*(np.max(v_tp_b)-np.min(v_tp_b))
    # # noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
    # # noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
    # # # u_tp_noise_b = u_tp_b + noise_u_b
    # # # v_tp_noise_b = v_tp_b + noise_v_b

    # Find params
    new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))
    new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
    # # # # # new_params_b = np.array(optimiser_b(u_tp_noise_b, v_tp_noise_b, dx_b))

    # Compute relative error and add them to matrix
    rel_error_mat_sch[i] = np.max(abs(new_params_sch - c_original_sch) / c_original_sch)
    rel_error_mat_fn[i] = np.max(abs(new_params_fn - c_original_fn) / c_original_fn)
    # # # # rel_error_mat_b[i] = np.max(abs(new_params_b-c_original_b)/c_original_b)

    rel_error_mat_mean_sch[i] = np.mean(
        abs(new_params_sch - c_original_sch) / c_original_sch
    )
    rel_error_mat_mean_fn[i] = np.mean(
        abs(new_params_fn - c_original_fn) / c_original_fn
    )
    # rel_error_mat_mean_b[i] = np.mean(
    # # # abs(new_params_b-c_original_b)/c_original_b)

    params_sch.append(new_params_sch)
    params_fn.append(new_params_fn)
    # # params_b.append(new_params_b)
    # Try to produce a new pattern
    # unew_tp_sch, vnew_tp_sch = pattern_S(new_params_sch)
    # unew_tp_fn, vnew_tp_fn = gen_pattern_fn(new_params_fn)
    # unew_tp_b, vnew_tp_b = gen_pattern_b(new_params_b)

    # # Compute MSE with original pattern and save
    # MSE_u_mat_sch[i] = np.mean((abs(np.fft.fft(unew_tp_sch)**2)-abs(np.fft.fft(u_tp_sch)**2)**2))
    # MSE_v_mat_sch[i] = np.mean((vnew_tp_sch-v_tp_sch)**2)
    # MSE_u_mat_fn[i] = np.mean((unew_tp_fn-u_tp_fn)**2)
    # MSE_v_mat_fn[i] = np.mean((vnew_tp_fn-v_tp_fn)**2)
    # MSE_u_mat_b[i] = np.mean((unew_tp_b-u_tp_b)**2)
    # MSE_v_mat_b[i] = np.mean((vnew_tp_b-v_tp_b)**2)


plt.plot(noise_array, rel_error_mat_sch)
plt.yscale("log")
plt.xscale("log")

plt.plot(noise_array, rel_error_mat_fn)
plt.yscale("log")
plt.xscale("log")

plt.plot(noise_array, rel_error_mat_b)
plt.yscale("log")
plt.xscale("log")

plt.plot(noise_array, rel_error_mat_mean_sch)
plt.yscale("log")
plt.xscale("log")

plt.plot(noise_array, rel_error_mat_mean_fn)
plt.yscale("log")
plt.xscale("log")

plt.plot(noise_array, rel_error_mat_mean_b)
plt.yscale("log")
plt.xscale("log")


# Will also make some new code to do same plots with different patterns


n_noise = 150
pattern_reps = 3
noise_array = np.geomspace(1e-6, 3, n_noise)
rel_error_mat_dif_pat_sch = np.zeros((n_noise, pattern_reps))
rel_error_mat_dif_pat_fn = np.zeros((n_noise, pattern_reps))
rel_error_mat_dif_pat_b = np.zeros((n_noise, pattern_reps))
rel_error_mat_dif_pat_mean_sch = np.zeros((n_noise, pattern_reps))
rel_error_mat_dif_pat_mean_fn = np.zeros((n_noise, pattern_reps))
rel_error_mat_dif_pat_mean_b = np.zeros((n_noise, pattern_reps))
params_sch = []
params_fn = []
params_b = []
patterns_u_sch = []
patterns_v_sch = []
patterns_u_fn = []
patterns_v_fn = []
patterns_u_b = []
patterns_v_b = []
for j in range(pattern_reps):
    n = 50
    c_S = [1, 40, 1, 0.1, 0.9, 1]  # S

    c_original_sch = [
        (c_S[3] * c_S[4]) / c_S[2] ** 2,
        c_S[5] * c_S[2] / (c_S[4] ** 2),
        c_S[1] / c_S[0],
        1,
    ]  # S nd
    u_tf = v_tf = c_S[4] / c_S[2]
    x_tf = np.sqrt(c_S[2] / c_S[0])
    t_tf = c_S[2]
    dx_sch = 1 * x_tf
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    perturbation1 = np.random.normal(0, 0.0001, (n**2))
    perturbation2 = np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u_tf * (u0 + perturbation1)
    y0[1::2] = v_tf * (v0 + perturbation2)
    tlen = int(90000)
    t = np.linspace(0, tlen, tlen)
    xrange1 = yrange1 = np.linspace(0, dx_sch * (n - 1), n)

    solb = odeint(
        step_forward_schnakenberg_nd3,
        y0,
        t,
        args=(c_original_sch, dx_sch),
        ml=2 * n,
        mu=2 * n,
    )
    u_tp_sch = np.reshape(solb[-1][::2], (n, n))
    v_tp_sch = np.reshape(solb[-1][1::2], (n, n))
    patterns_u_sch.append(u_tp_sch)
    patterns_v_sch.append(v_tp_sch)

    n = 50
    c_FN = [0.05, 0.00028, 1, 10, 1]  # FN du dv alpha eps mu
    c_original_fn = [c_FN[2], c_FN[1] / c_FN[0], c_FN[4] / c_FN[3], 1 / c_FN[3]]
    u_tf = v_tf = 1 / np.sqrt(c_FN[3])
    x_tf = 1 / np.sqrt(c_FN[0] / c_FN[3])
    t_tf = c_FN[3]
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    perturbation1 = u_tf * np.random.normal(0, 0.0001, (n**2))
    perturbation2 = v_tf * np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = c_S[4] * c_S[5] / c_S[0] ** 2 * (u0 + perturbation1)
    y0[1::2] = c_S[0] / c_S[4] * (v0 + perturbation2)
    tlen = int(90000)
    t = np.linspace(0, tlen, tlen)

    dx_fn = 1 / n * x_tf
    xrange1 = yrange1 = np.linspace(0, dx_fn * (n - 1), n)

    solb = odeint(
        step_forward_FN_nd, y0, t, args=(c_original_fn, dx_fn), ml=2 * n, mu=2 * n
    )
    u_tp_fn = np.reshape(solb[-1][::2], (n, n))
    v_tp_fn = np.reshape(solb[-1][1::2], (n, n))
    patterns_u_fn.append(u_tp_fn)
    patterns_v_fn.append(v_tp_fn)

    n = 50
    c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
    c_original_b = [
        c_B[0] / c_B[1],
        c_B[2] ** 2 / (c_B[3] + 1) ** 3,
        c_B[3] / (c_B[3] + 1),
    ]
    u_tf = v_tf = (c_B[3] - 1) / c_B[2]
    x_tf = 1 / np.sqrt(c_B[1] / (c_B[3] + 1))
    t_tf = c_B[3] + 1
    # Get initial conditions
    u0 = 0.1 * np.ones(n**2)
    v0 = 0.1 * np.ones(n**2)
    perturbation1 = u_tf * np.random.normal(0, 0.0001, (n**2))
    perturbation2 = v_tf * np.random.normal(0, 0.0001, (n**2))
    y0 = np.zeros(2 * n**2)
    y0[::2] = u0 + perturbation1
    y0[1::2] = v0 + perturbation2
    tlen = int(90000)
    t = np.linspace(0, tlen, tlen)

    dx_b = 1 / n * x_tf
    xrange1 = yrange1 = np.linspace(0, dx_b * (n - 1), n)

    solb = odeint(
        step_forward_B_nd, y0, t, args=(c_original_b, dx_b), ml=2 * n, mu=2 * n
    )
    u_tp_b = np.reshape(solb[-1][::2], (n, n))
    v_tp_b = np.reshape(solb[-1][1::2], (n, n))
    patterns_u_b.append(u_tp_b)
    patterns_v_b.append(v_tp_b)
    for i, noise_lev in enumerate(noise_array):
        # Add noise to patterns

        # Schnakenberg
        noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
        noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
        noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
        noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
        u_tp_noise_sch = u_tp_sch + noise_u_sch
        v_tp_noise_sch = v_tp_sch + noise_v_sch

        # FitzHugh-Nagumo
        noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
        noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
        noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
        noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
        u_tp_noise_fn = u_tp_fn + noise_u_fn
        v_tp_noise_fn = v_tp_fn + noise_v_fn

        # Brusselator
        noise_spread_u_b = noise_lev / 100 * (np.max(u_tp_b) - np.min(u_tp_b))
        noise_spread_v_b = noise_lev / 100 * (np.max(v_tp_b) - np.min(v_tp_b))
        noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
        noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
        u_tp_noise_b = u_tp_b + noise_u_b
        v_tp_noise_b = v_tp_b + noise_v_b

        # Find params
        new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))
        new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
        new_params_b = np.array(optimiser_b(u_tp_noise_b, v_tp_noise_b, dx_b))

        # Compute relative error and add them to matrix
        rel_error_mat_dif_pat_sch[i, j] = np.max(
            abs(new_params_sch - c_original_sch) / c_original_sch
        )
        rel_error_mat_dif_pat_fn[i, j] = np.max(
            abs(new_params_fn - c_original_fn) / c_original_fn
        )
        rel_error_mat_dif_pat_b[i, j] = np.max(
            abs(new_params_b - c_original_b) / c_original_b
        )

        rel_error_mat_dif_pat_mean_sch[i, j] = np.mean(
            abs(new_params_sch - c_original_sch) / c_original_sch
        )
        rel_error_mat_dif_pat_mean_fn[i, j] = np.mean(
            abs(new_params_fn - c_original_fn) / c_original_fn
        )
        rel_error_mat_dif_pat_mean_b[i, j] = np.mean(
            abs(new_params_b - c_original_b) / c_original_b
        )


plt.plot(noise_array[120:], rel_error_mat_dif_pat_sch[120:, :])
plt.yscale("log")
plt.xscale("log")

plt.plot(noise_array, rel_error_mat_dif_pat_fn)
plt.yscale("log")
plt.xscale("log")

plt.plot(noise_array, rel_error_mat_dif_pat_b)
plt.yscale("log")
plt.xscale("log")

plt.imshow(patterns_v_sch[2])

# We obtain a very similar result from each of the different patterns, but
# is this because of the patterns or because of the noise we add to the pattern?

# %% Run the noise plots

n_noise = 250
noise_reps = 30
noise_array = np.geomspace(1e-3, 3, n_noise)
rel_error_mat_noise_sch = np.zeros((n_noise, noise_reps))
rel_error_mat_noise_fn = np.zeros((n_noise, noise_reps))
rel_error_mat_noise_b = np.zeros((n_noise, noise_reps))
# rel_error_mat_noise_b_v2 = np.zeros((n_noise, noise_reps))
rel_error_mat_noise_mean_sch = np.zeros((n_noise, noise_reps))
rel_error_mat_noise_mean_fn = np.zeros((n_noise, noise_reps))
rel_error_mat_noise_mean_b = np.zeros((n_noise, noise_reps))
# rel_error_mat_noise_mean_b_v2 = np.zeros((n_noise, noise_reps))

rel_error_mat_noAbs_noise_sch = np.zeros((n_noise, noise_reps))
rel_error_mat_noAbs_noise_fn = np.zeros((n_noise, noise_reps))
rel_error_mat_noAbs_noise_b = np.zeros((n_noise, noise_reps))
# rel_error_mat_noAbs_noise_b_v2 = np.zeros((n_noise, noise_reps))
rel_error_mat_noAbs_noise_mean_sch = np.zeros((n_noise, noise_reps))
rel_error_mat_noAbs_noise_mean_fn = np.zeros((n_noise, noise_reps))
rel_error_mat_noAbs_noise_mean_b = np.zeros((n_noise, noise_reps))
# rel_error_mat_noAbs_noise_mean_b_v2 = np.zeros((n_noise, noise_reps))

# params_sch = []
# params_fn = []
# params_b = []
# patterns_u_sch = []
# patterns_v_sch = []
# patterns_u_fn = []
# patterns_v_fn = []
# patterns_u_b = []
# patterns_v_b = []
for j in range(noise_reps):
    for i, noise_lev in enumerate(noise_array):
        # Add noise to patterns

        # Schnakenberg
        noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
        noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
        noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
        noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
        u_tp_noise_sch = u_tp_sch + noise_u_sch
        v_tp_noise_sch = v_tp_sch + noise_v_sch

        # FitzHugh-Nagumo
        noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
        noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
        noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
        noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
        u_tp_noise_fn = u_tp_fn + noise_u_fn
        v_tp_noise_fn = v_tp_fn + noise_v_fn

        # Brusselator
        noise_spread_u_b = noise_lev / 100 * (np.max(u_tp_b) - np.min(u_tp_b))
        noise_spread_v_b = noise_lev / 100 * (np.max(v_tp_b) - np.min(v_tp_b))
        noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
        noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
        u_tp_noise_b = u_tp_b + noise_u_b
        v_tp_noise_b = v_tp_b + noise_v_b

        # Brusselator v2
        # noise_spread_u_b_v2 = noise_lev/100 * \
        # # (np.max(u_tp_b_v2)-np.min(u_tp_b_v2))
        # noise_spread_v_b_v2 = noise_lev/100 * \
        # # (np.max(v_tp_b_v2)-np.min(v_tp_b_v2))
        # # noise_u_b_v2 = np.random.normal(0, noise_spread_u_b_v2, (n, n))
        # # noise_v_b_v2 = np.random.normal(0, noise_spread_v_b_v2, (n, n))
        # # u_tp_noise_b_v2 = u_tp_b_v2 + noise_u_b
        # # v_tp_noise_b_v2 = v_tp_b_v2 + noise_v_b

        # Find params
        new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))
        new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
        new_params_b = np.array(optimiser_b(u_tp_noise_b, v_tp_noise_b, dx_b))
        # # new_params_b_v2 = np.array(optimiser_b_v2(
        # # # u_tp_noise_b_v2, v_tp_noise_b_v2, dx_b_v2))

        # Compute relative error and add them to matrix
        rel_error_mat_noise_sch[i, j] = np.max(
            abs(new_params_sch - c_original_sch) / c_original_sch
        )
        rel_error_mat_noise_fn[i, j] = np.max(
            abs(new_params_fn - c_original_fn) / c_original_fn
        )
        rel_error_mat_noise_b[i, j] = np.max(
            abs(new_params_b - c_original_b) / c_original_b
        )
        # rel_error_mat_noise_b_v2[i, j] = np.max(
        # # # abs(new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        rel_error_mat_noise_mean_sch[i, j] = np.mean(
            abs(new_params_sch - c_original_sch) / c_original_sch
        )
        rel_error_mat_noise_mean_fn[i, j] = np.mean(
            abs(new_params_fn - c_original_fn) / c_original_fn
        )
        rel_error_mat_noise_mean_b[i, j] = np.mean(
            abs(new_params_b - c_original_b) / c_original_b
        )
        # rel_error_mat_noise_mean_b_v2[i, j] = np.mean(
        # # # abs(new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        rel_error_mat_noAbs_noise_sch[i, j] = np.max(
            (new_params_sch - c_original_sch) / c_original_sch
        )
        rel_error_mat_noAbs_noise_fn[i, j] = np.max(
            (new_params_fn - c_original_fn) / c_original_fn
        )
        rel_error_mat_noAbs_noise_b[i, j] = np.max(
            (new_params_b - c_original_b) / c_original_b
        )
        # rel_error_mat_noAbs_noise_b_v2[i, j] = np.max(
        # # # (new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        rel_error_mat_noAbs_noise_mean_sch[i, j] = np.mean(
            (new_params_sch - c_original_sch) / c_original_sch
        )
        rel_error_mat_noAbs_noise_mean_fn[i, j] = np.mean(
            (new_params_fn - c_original_fn) / c_original_fn
        )
        rel_error_mat_noAbs_noise_mean_b[i, j] = np.mean(
            (new_params_b - c_original_b) / c_original_b
        )
        # rel_error_mat_noAbs_noise_mean_b_v2[i, j] = np.mean(
        # # # (new_params_b_v2-c_original_b_v2)/c_original_b_v2)
# %%


##########     FIGURE FOR PAPER      ##################
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
mean_sch = np.mean(rel_error_mat_noise_mean_sch, axis=1)
std_sch = np.std(rel_error_mat_noise_mean_sch, axis=1)
ax.plot(noise_array, mean_sch, color="tab:blue", label="Schnakenberg", alpha=0.6)
ax.fill_between(
    noise_array,
    mean_sch + std_sch,
    abs(mean_sch - std_sch),
    color="tab:blue",
    alpha=0.2,
)

mean_fn = np.mean(rel_error_mat_noise_mean_fn, axis=1)
std_fn = np.std(rel_error_mat_noise_mean_fn, axis=1)
ax.plot(noise_array, mean_fn, color="tab:orange", label="FitzHugh-Nagumo", alpha=0.6)
ax.fill_between(
    noise_array, mean_fn + std_fn, mean_fn - std_fn, color="tab:orange", alpha=0.2
)

mean_b = np.mean(rel_error_mat_noise_mean_b, axis=1)
std_b = np.std(rel_error_mat_noise_mean_b, axis=1)
ax.plot(noise_array, mean_b, color="tab:green", label="Brusselator")
ax.fill_between(
    noise_array, mean_b + std_b, abs(mean_b - std_b), color="tab:green", alpha=0.5
)
# mean_b_v2 = np.mean(rel_error_mat_noise_mean_b_v2, axis=1)
# std_b_v2 = np.std(rel_error_mat_noise_mean_b_v2, axis=1)
# ax.plot(noise_array, mean_b_v2, color='orange', label='Brusselator_v2')
# ax.fill_between(noise_array, mean_b_v2+std_b_v2,
#                 abs(mean_b_v2-std_b_v2), color='orange', alpha=0.5)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Relative Error (%)", fontsize=14)
ax.set_xlabel("Relative Noise (%)", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.tick_params(axis="both", which="minor", labelsize=16)
plt.legend(fontsize=14)
plt.savefig("Relative Noise vs Error all legend")
plt.savefig("Relative Noise vs Error all legend.pdf")
#######################            #########################

fig, ax = plt.subplots(1)
mean_sch = np.mean(rel_error_mat_noise_sch, axis=1)
std_sch = np.std(rel_error_mat_noise_sch, axis=1)
ax.plot(noise_array, mean_sch, color="blue", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="blue", alpha=0.5
)

mean_fn = np.mean(rel_error_mat_noise_fn, axis=1)
std_fn = np.std(rel_error_mat_noise_fn, axis=1)
ax.plot(noise_array, mean_fn, color="red", label="FitzHugh-Nagumo")
ax.fill_between(noise_array, mean_fn + std_fn, mean_fn - std_fn, color="red", alpha=0.5)

mean_b = np.mean(rel_error_mat_noise_b, axis=1)
std_b = np.std(rel_error_mat_noise_b, axis=1)
ax.plot(noise_array, mean_b, color="green", label="Brusselator")
ax.fill_between(
    noise_array, mean_b + std_b, abs(mean_b - std_b), color="green", alpha=0.5
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Relative Error (%)")
ax.set_xlabel("Relative Noise (%)")


fig, ax = plt.subplots(1)
mean_sch = np.mean(rel_error_mat_noAbs_noise_sch, axis=1)
std_sch = np.std(rel_error_mat_noAbs_noise_sch, axis=1)
ax.plot(noise_array, mean_sch, color="royalblue", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="royalblue", alpha=0.5
)

mean_fn = np.mean(rel_error_mat_noAbs_noise_fn, axis=1)
std_fn = np.std(rel_error_mat_noAbs_noise_fn, axis=1)
ax.plot(noise_array, mean_fn, color="red", label="FitzHugh-Nagumo")
ax.fill_between(noise_array, mean_fn + std_fn, mean_fn - std_fn, color="red", alpha=0.5)

mean_b = np.mean(rel_error_mat_noAbs_noise_b, axis=1)
std_b = np.std(rel_error_mat_noAbs_noise_b, axis=1)
ax.plot(noise_array, mean_b, color="green", label="Brusselator")
ax.fill_between(
    noise_array, mean_b + std_b, abs(mean_b - std_b), color="green", alpha=0.5
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Relative Error (%)")
ax.set_xlabel("Relative Noise (%)")
plt.savefig("Difference_params_vs_rel_noise.pdf")


fig, ax = plt.subplots(1)
mean_sch = np.mean(rel_error_mat_noAbs_noise_mean_sch, axis=1)
std_sch = np.std(rel_error_mat_noAbs_noise_mean_sch, axis=1)
ax.plot(noise_array, -mean_sch, color="royalblue", label="Negative Values")
ax.fill_between(
    noise_array, -mean_sch + std_sch, -mean_sch - std_sch, color="royalblue", alpha=0.5
)

ax.plot(noise_array, mean_sch, color="orange", label="Positive Values")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="orange", alpha=0.5
)
plt.legend()
# mean_fn = np.mean(rel_error_mat_noAbs_noise_mean_fn, axis=1)
# std_fn = np.std(rel_error_mat_noAbs_noise_mean_fn, axis=1)
# ax.plot(noise_array, mean_fn, color='yellow', label='FitzHugh-Nagumo')
# ax.fill_between(noise_array,mean_fn+std_fn,mean_fn-std_fn, color='yellow', alpha = 0.5)

# ax.plot(noise_array, -mean_fn, color='red', label='FitzHugh-Nagumo')
# ax.fill_between(noise_array,-mean_fn+std_fn,-mean_fn-std_fn, color='red', alpha = 0.5)

# mean_b = np.mean(rel_error_mat_noAbs_noise_mean_b, axis=1)
# std_b = np.std(rel_error_mat_noAbs_noise_mean_b, axis=1)
# ax.plot(noise_array, mean_b, color='green', label='Brusselator')
# ax.fill_between(noise_array,mean_b+std_b,mean_b-std_b, color='green', alpha = 0.5)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Parameter Difference (%)")
ax.set_xlabel("Relative Noise (%)")
plt.savefig("Difference_params_vs_rel_noise.pdf")
# ax.legend()

# Increase the amount of replications, increase range slightly and decrease amount of points
plt.legend()
plt.plot(std_sch)
plt.plot(std_fn)
plt.plot(std_b)
plt.yscale("log")
plt.legend(["a", "b", "c"])


n = 50
c_S = [1, 40, 1, 0.1, 0.9, 1]  # S

c_original_sch = [
    (c_S[3] * c_S[4]) / c_S[2] ** 2,
    c_S[5] * c_S[2] / (c_S[4] ** 2),
    c_S[1] / c_S[0],
    1,
]  # S nd
u_tf = v_tf = c_S[4] / c_S[2]
x_tf = np.sqrt(c_S[2] / c_S[0])
t_tf = c_S[2]
dx_sch = 1 * x_tf
u0 = 0.1 * np.ones(n**2)
v0 = 0.1 * np.ones(n**2)
perturbation1 = np.random.normal(0, 0.0001, (n**2))
perturbation2 = np.random.normal(0, 0.0001, (n**2))
y0 = np.zeros(2 * n**2)
y0[::2] = u_tf * (u0 + perturbation1)
y0[1::2] = v_tf * (v0 + perturbation2)
tlen = int(200000)
t = np.linspace(0, tlen, tlen)
xrange1 = yrange1 = np.linspace(0, dx_sch * (n - 1), n)

solb = odeint(
    step_forward_schnakenberg_nd3,
    y0,
    t,
    args=(c_original_sch, dx_sch),
    ml=2 * n,
    mu=2 * n,
)
u_tp_sch_1 = np.reshape(solb[-1][::2], (n, n))
v_tp_sch_1 = np.reshape(solb[-1][1::2], (n, n))

perturbation1 = np.random.normal(0, 0.0001, (n**2))
perturbation2 = np.random.normal(0, 0.0001, (n**2))
y0 = np.zeros(2 * n**2)
y0[::2] = u_tf * (u0 + perturbation1)
y0[1::2] = v_tf * (v0 + perturbation2)
solb = odeint(
    step_forward_schnakenberg_nd3,
    y0,
    t,
    args=(c_original_sch, dx_sch),
    ml=2 * n,
    mu=2 * n,
)
u_tp_sch_2 = np.reshape(solb[-1][::2], (n, n))
v_tp_sch_2 = np.reshape(solb[-1][1::2], (n, n))

c_sch_2 = [
    (c_S[3] * c_S[4]) / c_S[2] ** 2 + 0.04,
    c_S[5] * c_S[2] / (c_S[4] ** 2),
    c_S[1] / c_S[0] + 1,
    1 + 0.4,
]
solb = odeint(
    step_forward_schnakenberg_nd3, y0, t, args=(c_sch_2, dx_sch), ml=2 * n, mu=2 * n
)
u_tp_sch_3 = np.reshape(solb[-1][::2], (n, n))
v_tp_sch_3 = np.reshape(solb[-1][1::2], (n, n))
# We will now see how the parameters are localised at three different points compared to the original parameters


# %% Run the noise plots


n_noise = 20
noise_reps = 40
noise_array = np.geomspace(1e-4, 1e-1, n_noise)

params_sch = []
params_fn = []
params_b = []
patterns_u_sch = []
patterns_v_sch = []
patterns_u_fn = []
patterns_v_fn = []
patterns_u_b = []
patterns_v_b = []

for noise_lev in noise_array:
    sch_points = np.zeros((4, noise_reps))
    fn_points = np.zeros((4, noise_reps))
    b_points = np.zeros((3, noise_reps))
    b_v2_points = np.zeros((3, noise_reps))
    for j in range(noise_reps):
        # Add noise to patterns

        # Schnakenberg
        noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
        noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
        noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
        noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
        u_tp_noise_sch = u_tp_sch + noise_u_sch
        v_tp_noise_sch = v_tp_sch + noise_v_sch

        # FitzHugh-Nagumo
        noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
        noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
        noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
        noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
        u_tp_noise_fn = u_tp_fn + noise_u_fn
        v_tp_noise_fn = v_tp_fn + noise_v_fn

        # Brusselator
        noise_spread_u_b = noise_lev / 100 * (np.max(u_tp_b) - np.min(u_tp_b))
        noise_spread_v_b = noise_lev / 100 * (np.max(v_tp_b) - np.min(v_tp_b))
        noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
        noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
        u_tp_noise_b = u_tp_b + noise_u_b
        v_tp_noise_b = v_tp_b + noise_v_b

        # Brusselator v2
        noise_spread_u_b_v2 = noise_lev / 100 * (np.max(u_tp_b_v2) - np.min(u_tp_b_v2))
        noise_spread_v_b_v2 = noise_lev / 100 * (np.max(v_tp_b_v2) - np.min(v_tp_b_v2))
        noise_u_b_v2 = np.random.normal(0, noise_spread_u_b_v2, (n, n))
        noise_v_b_v2 = np.random.normal(0, noise_spread_v_b_v2, (n, n))
        u_tp_noise_b_v2 = u_tp_b_v2 + noise_u_b
        v_tp_noise_b_v2 = v_tp_b_v2 + noise_v_b

        # Find params
        new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))
        new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
        new_params_b = np.array(optimiser_b(u_tp_noise_b, v_tp_noise_b, dx_b))
        new_params_b_v2 = np.array(
            optimiser_b_v2(u_tp_noise_b_v2, v_tp_noise_b_v2, dx_b_v2)
        )

        sch_points[:, j] = new_params_sch
        fn_points[:, j] = new_params_fn
        b_points[:, j] = new_params_b
        b_v2_points[:, j] = new_params_b_v2

    # Adding original parameters
    print(noise_lev)
    # plt.scatter(sch_points[0, :], sch_points[1, :])
    # plt.scatter(c_original_sch[0], c_original_sch[1])
    # plt.show()
    # plt.scatter(sch_points[2, :], sch_points[3, :])
    # plt.scatter(c_original_sch[2], c_original_sch[3])
    # plt.show()
    plt.scatter(fn_points[0, :], fn_points[1, :])
    plt.scatter(c_original_fn[0], c_original_fn[1])
    plt.show()
    plt.scatter(fn_points[2, :], fn_points[3, :])
    plt.scatter(c_original_fn[2], c_original_fn[3])
    plt.show()


# We will try and repeat the previous plot with different patterns, so will see how
#  the plots depend on the pattern


# %% Run the noise plots with different patterns
n_noise = 250
noise_reps = 30
n_patterns = 5
noise_array = np.geomspace(1e-4, 10, n_noise)
rel_error_mat_dif_pat_noise_sch = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noise_fn = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noise_b = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noise_b_v2 = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noise_mean_sch = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noise_mean_fn = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noise_mean_b = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noise_mean_b_v2 = np.zeros((n_noise, noise_reps, n_patterns))

rel_error_mat_dif_pat_noAbs_noise_sch = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noAbs_noise_fn = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noAbs_noise_b = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noAbs_noise_b_v2 = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noAbs_noise_mean_sch = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noAbs_noise_mean_fn = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noAbs_noise_mean_b = np.zeros((n_noise, noise_reps, n_patterns))
rel_error_mat_dif_pat_noAbs_noise_mean_b_v2 = np.zeros(
    (n_noise, noise_reps, n_patterns)
)

params_sch = []
params_fn = []
params_b = []

patterns_sch_mat = []
patterns_fn_mat = []
patterns_b_mat = []
patterns_b_v2_mat = []
for m in range(n_patterns):
    # Get patterns
    n = 50

    c_S = [1, 40, 1, 0.1, 0.9, 1]  # S
    c_original_sch = [
        (c_S[3] * c_S[4]) / c_S[2] ** 2,
        c_S[5] * c_S[2] / (c_S[4] ** 2),
        c_S[1] / c_S[0],
        1,
    ]  # S nd
    u_tf = v_tf = c_S[4] / c_S[2]
    x_tf = np.sqrt(c_S[2] / c_S[0])
    dx_sch = 1 * x_tf

    c_FN = [0.05, 0.00028, 1, 10, 1]  # FN du dv alpha eps mu
    c_original_fn = [c_FN[2], c_FN[1] / c_FN[0], c_FN[4] / c_FN[3], 1 / c_FN[3]]
    u_tf = v_tf = 1 / np.sqrt(c_FN[3])
    x_tf = 1 / np.sqrt(c_FN[0] / c_FN[3])
    dx_fn = 1 / n * x_tf

    c_B = [0.0016, 0.0131, 4.5, 6.96]  # B
    c_original_b = [
        c_B[0] / c_B[1],
        c_B[2] ** 2 / (c_B[3] + 1) ** 3,
        c_B[3] / (c_B[3] + 1),
    ]
    u_tf = v_tf = (c_B[3] - 1) / c_B[2]
    x_tf = 1 / np.sqrt(c_B[1] / (c_B[3] + 1))
    dx_b = 1 / n * x_tf

    c_original_b_v2 = [
        (c_B[3] + 1) / (c_B[3]),
        c_B[2] ** 2 / (c_B[3]) ** 3,
        c_B[1] / c_B[0],
    ]
    u_tf = v_tf = (c_B[3]) / c_B[2]
    x_tf = 1 / np.sqrt(c_B[0] / (c_B[3]))
    dx_b_v2 = 1 / n * x_tf

    u_tp_sch, v_tp_sch = gen_pattern_sch(c_original_sch)
    u_tp_fn, v_tp_fn = gen_pattern_fn(c_original_fn)
    # u_tp_b, v_tp_b = patterns_b_mat[1]
    u_tp_b, v_tp_b = gen_pattern_b(c_original_b)
    u_tp_b_v2, v_tp_b_v2 = gen_pattern_b_v2(c_original_b_v2)
    patterns_sch_mat.append([u_tp_sch, v_tp_sch])
    patterns_fn_mat.append([u_tp_fn, v_tp_fn])
    patterns_b_mat.append([u_tp_b, v_tp_b])
    patterns_b_v2_mat.append([u_tp_b_v2, v_tp_b_v2])
    # noise_lev = 0
    for j in range(noise_reps):
        for i, noise_lev in enumerate(noise_array):
            # Add noise to patterns

            # Schnakenberg
            noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
            noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
            noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
            noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
            u_tp_noise_sch = u_tp_sch + noise_u_sch
            v_tp_noise_sch = v_tp_sch + noise_v_sch

            # FitzHugh-Nagumo
            noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
            noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
            noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
            noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
            u_tp_noise_fn = u_tp_fn + noise_u_fn
            v_tp_noise_fn = v_tp_fn + noise_v_fn

            # Brusselator
            noise_spread_u_b = noise_lev / 100 * (np.max(u_tp_b) - np.min(u_tp_b))
            noise_spread_v_b = noise_lev / 100 * (np.max(v_tp_b) - np.min(v_tp_b))
            noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
            noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
            u_tp_noise_b = u_tp_b + noise_u_b
            v_tp_noise_b = v_tp_b + noise_v_b

            # Brusselator v2
            noise_spread_u_b_v2 = (
                noise_lev / 100 * (np.max(u_tp_b_v2) - np.min(u_tp_b_v2))
            )
            noise_spread_v_b_v2 = (
                noise_lev / 100 * (np.max(v_tp_b_v2) - np.min(v_tp_b_v2))
            )
            noise_u_b_v2 = np.random.normal(0, noise_spread_u_b_v2, (n, n))
            noise_v_b_v2 = np.random.normal(0, noise_spread_v_b_v2, (n, n))
            u_tp_noise_b_v2 = u_tp_b_v2 + noise_u_b_v2
            v_tp_noise_b_v2 = v_tp_b_v2 + noise_v_b_v2

            # Find params
            new_params_sch = np.array(
                optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch)
            )
            new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
            new_params_b = np.array(optimiser_b(u_tp_noise_b, v_tp_noise_b, dx_b))
            new_params_b_v2 = np.array(
                optimiser_b_v2(u_tp_noise_b_v2, v_tp_noise_b_v2, dx_b_v2)
            )

            # Compute relative error and add them to matrix
            rel_error_mat_dif_pat_noise_sch[i, j, m] = np.max(
                abs(new_params_sch - c_original_sch) / c_original_sch
            )
            rel_error_mat_dif_pat_noise_fn[i, j, m] = np.max(
                abs(new_params_fn - c_original_fn) / c_original_fn
            )
            rel_error_mat_dif_pat_noise_b[i, j, m] = np.max(
                abs(new_params_b - c_original_b) / c_original_b
            )
            rel_error_mat_dif_pat_noise_b_v2[i, j, m] = np.max(
                abs(new_params_b_v2 - c_original_b_v2) / c_original_b_v2
            )

            rel_error_mat_dif_pat_noise_mean_sch[i, j, m] = np.mean(
                abs(new_params_sch - c_original_sch) / c_original_sch
            )
            rel_error_mat_dif_pat_noise_mean_fn[i, j, m] = np.mean(
                abs(new_params_fn - c_original_fn) / c_original_fn
            )
            rel_error_mat_dif_pat_noise_mean_b[i, j, m] = np.mean(
                abs(new_params_b - c_original_b) / c_original_b
            )
            rel_error_mat_dif_pat_noise_mean_b_v2[i, j, m] = np.mean(
                abs(new_params_b_v2 - c_original_b_v2) / c_original_b_v2
            )

            rel_error_mat_dif_pat_noAbs_noise_sch[i, j, m] = np.max(
                (new_params_sch - c_original_sch) / c_original_sch
            )
            rel_error_mat_dif_pat_noAbs_noise_fn[i, j, m] = np.max(
                (new_params_fn - c_original_fn) / c_original_fn
            )
            rel_error_mat_dif_pat_noAbs_noise_b[i, j, m] = np.max(
                (new_params_b - c_original_b) / c_original_b
            )
            rel_error_mat_dif_pat_noAbs_noise_b_v2[i, j, m] = np.max(
                (new_params_b_v2 - c_original_b_v2) / c_original_b_v2
            )

            rel_error_mat_dif_pat_noAbs_noise_mean_sch[i, j, m] = np.mean(
                (new_params_sch - c_original_sch) / c_original_sch
            )
            rel_error_mat_dif_pat_noAbs_noise_mean_fn[i, j, m] = np.mean(
                (new_params_fn - c_original_fn) / c_original_fn
            )
            rel_error_mat_dif_pat_noAbs_noise_mean_b[i, j, m] = np.mean(
                (new_params_b - c_original_b) / c_original_b
            )
            rel_error_mat_dif_pat_noAbs_noise_mean_b_v2[i, j, m] = np.mean(
                (new_params_b_v2 - c_original_b_v2) / c_original_b_v2
            )
# %%
# now we plot these for each of the model
# Schnakenberg
fig, ax = plt.subplots(1)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_sch[:, :, 0], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_sch[:, :, 0], axis=1)
ax.plot(noise_array, mean_sch, color="blue", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="blue", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_sch[:, :, 1], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_sch[:, :, 1], axis=1)
ax.plot(noise_array, mean_sch, color="teal", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="teal", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_sch[:, :, 2], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_sch[:, :, 2], axis=1)
ax.plot(noise_array, mean_sch, color="cyan", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="cyan", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_sch[:, :, 3], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_sch[:, :, 3], axis=1)
ax.plot(noise_array, mean_sch, color="darkcyan", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="darkcyan", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_sch[:, :, 4], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_sch[:, :, 4], axis=1)
ax.plot(noise_array, mean_sch, color="purple", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="purple", alpha=0.1
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Relative Error (%)")
ax.set_xlabel("Relative Noise (%)")


plt.plot(noise_array, std_sch)
plt.yscale("log")
plt.xscale("log")
plt.plot(noise_array, std_fn)
plt.yscale("log")
plt.xscale("log")
plt.plot(noise_array, mean_sch)
plt.yscale("log")
plt.xscale("log")
plt.plot(noise_array, mean_fn)
plt.yscale("log")
plt.xscale("log")
# FitzHugh-Nagumo
plt.imshow(patterns_b_mat[2][0])
plt.colorbar()

fig, ax = plt.subplots(1)
mean_fn = np.mean(rel_error_mat_dif_pat_noise_mean_fn[:, :, 0], axis=1)
std_fn = np.std(rel_error_mat_dif_pat_noise_mean_fn[:, :, 0], axis=1)
ax.plot(noise_array, mean_fn, color="blue", label="fnnakenberg")
ax.fill_between(
    noise_array, mean_fn + std_fn, mean_fn - std_fn, color="blue", alpha=0.1
)
mean_fn = np.mean(rel_error_mat_dif_pat_noise_mean_fn[:, :, 1], axis=1)
std_fn = np.std(rel_error_mat_dif_pat_noise_mean_fn[:, :, 1], axis=1)
ax.plot(noise_array, mean_fn, color="teal", label="fnnakenberg")
ax.fill_between(
    noise_array, mean_fn + std_fn, mean_fn - std_fn, color="teal", alpha=0.1
)
mean_fn = np.mean(rel_error_mat_dif_pat_noise_mean_fn[:, :, 2], axis=1)
std_fn = np.std(rel_error_mat_dif_pat_noise_mean_fn[:, :, 2], axis=1)
ax.plot(noise_array, mean_fn, color="cyan", label="fnnakenberg")
ax.fill_between(
    noise_array, mean_fn + std_fn, mean_fn - std_fn, color="cyan", alpha=0.1
)
mean_fn = np.mean(rel_error_mat_dif_pat_noise_mean_fn[:, :, 3], axis=1)
std_fn = np.std(rel_error_mat_dif_pat_noise_mean_fn[:, :, 3], axis=1)
ax.plot(noise_array, mean_fn, color="darkcyan", label="fnnakenberg")
ax.fill_between(
    noise_array, mean_fn + std_fn, mean_fn - std_fn, color="darkcyan", alpha=0.1
)
mean_fn = np.mean(rel_error_mat_dif_pat_noise_mean_fn[:, :, 4], axis=1)
std_fn = np.std(rel_error_mat_dif_pat_noise_mean_fn[:, :, 4], axis=1)
ax.plot(noise_array, mean_fn, color="purple", label="fnnakenberg")
ax.fill_between(
    noise_array, mean_fn + std_fn, mean_fn - std_fn, color="purple", alpha=0.1
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Relative Error (%)")
ax.set_xlabel("Relative Noise (%)")


# Brusselator
fig, ax = plt.subplots(1)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b[:, :, 0], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b[:, :, 0], axis=1)
ax.plot(noise_array, mean_sch, color="blue", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="blue", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b[:, :, 1], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b[:, :, 1], axis=1)
ax.plot(noise_array, mean_sch, color="teal", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="teal", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b[:, :, 2], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b[:, :, 2], axis=1)
ax.plot(noise_array, mean_sch, color="cyan", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="cyan", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b[:, :, 3], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b[:, :, 3], axis=1)
ax.plot(noise_array, mean_sch, color="darkcyan", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="darkcyan", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b[:, :, 4], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b[:, :, 4], axis=1)
ax.plot(noise_array, mean_sch, color="purple", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="purple", alpha=0.1
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Relative Error (%)")
ax.set_xlabel("Relative Noise (%)")
# Brusselator v2


fig, ax = plt.subplots(1)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 0], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 0], axis=1)
ax.plot(noise_array, mean_sch, color="blue", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="blue", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 1], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 1], axis=1)
ax.plot(noise_array, mean_sch, color="teal", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="teal", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 2], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 2], axis=1)
ax.plot(noise_array, mean_sch, color="cyan", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="cyan", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 3], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 3], axis=1)
ax.plot(noise_array, mean_sch, color="darkcyan", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="darkcyan", alpha=0.1
)
mean_sch = np.mean(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 4], axis=1)
std_sch = np.std(rel_error_mat_dif_pat_noise_mean_b_v2[:, :, 4], axis=1)
ax.plot(noise_array, mean_sch, color="purple", label="Schnakenberg")
ax.fill_between(
    noise_array, mean_sch + std_sch, mean_sch - std_sch, color="purple", alpha=0.1
)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Relative Error (%)")
ax.set_xlabel("Relative Noise (%)")


plt.imshow(u_tp_sch_1)
plt.imshow(u_tp_sch_3)
plt.imshow(v_tp_b_v2)
cmap = cm.Spectral
plt.imshow(
    (np.fft.fft2(u_tp_sch_1).real ** 2 + np.fft.fft2(u_tp_sch_1).imag ** 2)[1:, 1:],
    cmap=cmap,
)
plt.colorbar()
plt.imshow(
    (np.fft.fft2(u_tp_sch_2).real ** 2 + np.fft.fft2(u_tp_sch_2).imag ** 2)[1:, 1:],
    cmap=cmap,
)
plt.colorbar()
plt.imshow((np.abs(np.fft.fftshift(np.fft.fft2(u_tp_sch_1)[1:, 1:])) ** 2), cmap=cmap)
plt.colorbar()
plt.imshow((np.abs(np.fft.fftshift(np.fft.fft2(u_tp_sch_2)[1:, 1:])) ** 2), cmap=cmap)
plt.colorbar()
image1 = np.abs(np.fft.fftshift(np.fft.fft2(u_tp_sch_1)[1:, 1:])) ** 2
image2 = np.abs(np.fft.fftshift(np.fft.fft2(u_tp_sch_2)[1:, 1:])) ** 2
image3 = np.abs(np.fft.fftshift(np.fft.fft2(v_tp_b_v2)[1:, 1:])) ** 2
image4 = np.abs(np.fft.fftshift(np.fft.fft2(u_tp_sch_3)[1:, 1:])) ** 2
x, y = np.meshgrid(np.arange(image1.shape[1]), np.arange(image1.shape[0]))
R = np.sqrt(x**2 + y**2)

# calculate the mean
plt.imshow(image1)
plt.imshow(image2)
plt.imshow(image3)
plt.imshow(image4)


def f1(r):
    return image1[(R >= r - 2) & (R < r + 2)].mean()


r1 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
mean1 = np.vectorize(f1)(r1)


def f2(r):
    return image2[(R >= r - 2) & (R < r + 2)].mean()


r2 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
mean2 = np.vectorize(f2)(r2)


def f3(r):
    return image3[(R >= r - 2) & (R < r + 2)].mean()


r3 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
mean3 = np.vectorize(f3)(r3)


def f4(r):
    return image4[(R >= r - 2) & (R < r + 2)].mean()


r4 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
mean4 = np.vectorize(f4)(r4)

# plot it
fig, ax = plt.subplots()
ax.plot(r1, mean1)
ax.plot(r2, mean2)
ax.plot(r3, mean3)
ax.plot(r4, mean4)
plt.show()

np.mean(mean1 - mean2) / np.mean(mean1)
np.mean(mean1 - mean3) / np.mean(mean1)
np.mean(mean1 - mean4) / np.mean(mean1)


# Let's do one part for the plot of the Fourier

# First write some code to compare the spectral radial amplitude of two images


def spectral_power_radial(tp_new, tp_or, plot=False):
    """
    Spectral power averaged with the radius for new and old TP
    """
    tp_sp_p_1 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new)[1:, 1:])) ** 2
    tp_sp_p_2 = np.abs(np.fft.fftshift(np.fft.fft2(tp_or)[1:, 1:])) ** 2
    x, y = np.meshgrid(np.arange(tp_sp_p_1.shape[1]), np.arange(tp_sp_p_1.shape[0]))
    R = np.sqrt(x**2 + y**2)

    def f1(r):
        return tp_sp_p_1[(R >= r - 2) & (R < r + 2)].mean()

    r1 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
    mean1 = np.vectorize(f1)(r1)

    def f2(r):
        return tp_sp_p_2[(R >= r - 2) & (R < r + 2)].mean()

    r2 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
    mean2 = np.vectorize(f2)(r2)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(r1, mean1)
        ax.plot(r2, mean2)
        plt.show()

    return np.mean(abs(mean1 - mean2)) / np.mean(abs(mean2))


def spectral_power_radial_norm(tp_new, tp_or, plot=False):
    """
    Spectral power averaged with the radius for new and old TP
    """
    tp_new = (tp_new - np.min(tp_new)) / (np.max(tp_new) - np.min(tp_new))
    tp_or = (tp_or - np.min(tp_or)) / (np.max(tp_or) - np.min(tp_or))
    tp_sp_p_1 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new)[1:, 1:])) ** 2
    tp_sp_p_2 = np.abs(np.fft.fftshift(np.fft.fft2(tp_or)[1:, 1:])) ** 2
    x, y = np.meshgrid(np.arange(tp_sp_p_1.shape[1]), np.arange(tp_sp_p_1.shape[0]))
    R = np.sqrt(x**2 + y**2)

    def f1(r):
        return tp_sp_p_1[(R >= r - 2) & (R < r + 2)].mean()

    r1 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
    mean1 = np.vectorize(f1)(r1)

    def f2(r):
        return tp_sp_p_2[(R >= r - 2) & (R < r + 2)].mean()

    r2 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
    mean2 = np.vectorize(f2)(r2)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(r1, mean1)
        ax.plot(r2, mean2)
        plt.show()

    return np.mean(abs(mean1 - mean2)) / np.mean(abs(mean2))


def spectral_power_radial_norm_v2(tp_new, tp_or, plot=False):
    """
    Spectral power averaged with the radius for new and old TP
    """

    tp_new = (tp_new - np.min(tp_new)) / (np.max(tp_or) - np.min(tp_or))
    tp_or = (tp_or - np.min(tp_or)) / (np.max(tp_or) - np.min(tp_or))
    tp_sp_p_1 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new)[1:, 1:])) ** 2
    tp_sp_p_2 = np.abs(np.fft.fftshift(np.fft.fft2(tp_or)[1:, 1:])) ** 2
    x, y = np.meshgrid(np.arange(tp_sp_p_1.shape[1]), np.arange(tp_sp_p_1.shape[0]))
    R = np.sqrt(x**2 + y**2)

    def f1(r):
        return tp_sp_p_1[(R >= r - 2) & (R < r + 2)].mean()

    r1 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
    mean1 = np.vectorize(f1)(r1)

    def f2(r):
        return tp_sp_p_2[(R >= r - 2) & (R < r + 2)].mean()

    r2 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
    mean2 = np.vectorize(f2)(r2)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(r1, mean1)
        ax.plot(r2, mean2)
        plt.show()

    return np.mean(abs(mean1 - mean2)) / np.mean(abs(mean2))


def spectral_power_radial_norm(tp_new, tp_or, plot=False):
    """
    Spectral power averaged with the radius for new and old TP
    """
    tp_new = (tp_new - np.min(tp_new)) / (np.max(tp_new) - np.min(tp_new))
    tp_or = (tp_or - np.min(tp_or)) / (np.max(tp_or) - np.min(tp_or))
    tp_sp_p_1 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new))) ** 2
    tp_sp_p_2 = np.abs(np.fft.fftshift(np.fft.fft2(tp_or))) ** 2
    x, y = np.meshgrid(np.arange(tp_sp_p_1.shape[1]), np.arange(tp_sp_p_1.shape[0]))
    inverse = np.linspace(tp_sp_p_1.shape[1] // 2, 1, tp_sp_p_1.shape[1] // 2).astype(
        int
    )
    x[:, -tp_sp_p_1.shape[1] // 2 :] = inverse
    y = x.T
    R = np.fft.fftshift(np.sqrt(x**2 + y**2))

    def f1(r):
        return tp_sp_p_1[(R >= r - 0.7) & (R < r + 0.7)].mean()

    r1 = np.linspace(1.5, int(np.max(R)), num=3 * int(np.max(R)) - 2)
    mean1 = np.vectorize(f1)(r1)

    def f2(r):
        return tp_sp_p_2[(R >= r - 0.7) & (R < r + 0.7)].mean()

    r2 = np.linspace(1.5, int(np.max(R)), num=3 * int(np.max(R)) - 2)
    mean2 = np.vectorize(f2)(r2)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(r1, mean1)
        ax.plot(r2, mean2)
        plt.show()
    return np.mean(abs(mean1 - mean2)) / np.mean(abs(mean2))


# We want to repeat what we had before but with the Fourier plot too
# %% Run the noise plots
n_noise = 100
noise_reps = 25
noise_array = np.geomspace(1e-2, 4, n_noise)

# noise_array_2 = np.geomspace(1.01, 4, 25)
# rel_error_mat_fourier_noise_sch = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_fn = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_b = np.zeros((n_noise, noise_reps))
# # rel_error_mat_fourier_noise_b_v2 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_mean_sch = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_mean_fn = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_mean_b = np.zeros((n_noise, noise_reps))
# # rel_error_mat_fourier_noise_mean_b_v2 = np.zeros((n_noise, noise_reps))

# rel_error_mat_fourier_noAbs_noise_sch = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_fn = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_b = np.zeros((n_noise, noise_reps))
# # rel_error_mat_fourier_noAbs_noise_b_v2 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_mean_sch = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_mean_fn = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_mean_b = np.zeros((n_noise, noise_reps))
# # rel_error_mat_fourier_noAbs_noise_mean_b_v2 = np.zeros((n_noise, noise_reps))

params_sch = []
params_fn = []
params_b = []
fourier_u_sch = []
fourier_v_sch = []
fourier_u_fn = []
fourier_v_fn = []
fourier_u_b = []
fourier_v_b = []
# fourier_u_b_v2 = []
# fourier_v_b_v2 = []
fourier_u_sch_norm = []
fourier_v_sch_norm = []
fourier_u_fn_norm = []
fourier_v_fn_norm = []
fourier_u_b_norm = []
fourier_v_b_norm = []
# fourier_u_b_v2_norm = []
# fourier_v_b_v2_norm = []
noise_lev = 0
for j in range(noise_reps):
    for i, noise_lev in enumerate(noise_array):
        # Add noise to patterns

        # Schnakenberg
        noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
        noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
        noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
        noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
        u_tp_noise_sch = u_tp_sch + noise_u_sch
        v_tp_noise_sch = v_tp_sch + noise_v_sch

        # FitzHugh-Nagumo
        noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
        noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
        noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
        noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
        u_tp_noise_fn = u_tp_fn + noise_u_fn
        v_tp_noise_fn = v_tp_fn + noise_v_fn

        # Brusselator
        noise_spread_u_b = noise_lev / 100 * (np.max(u_tp_b) - np.min(u_tp_b))
        noise_spread_v_b = noise_lev / 100 * (np.max(v_tp_b) - np.min(v_tp_b))
        noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
        noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
        u_tp_noise_b = u_tp_b + noise_u_b
        v_tp_noise_b = v_tp_b + noise_v_b

        # Brusselator v2
        # # noise_spread_u_b = noise_lev/100*(np.max(u_tp_b_v2)-np.min(u_tp_b_v2))
        # # noise_spread_v_b = noise_lev/100*(np.max(v_tp_b_v2)-np.min(v_tp_b_v2))
        # noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
        # noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
        # # u_tp_noise_b_v2 = u_tp_b_v2 + noise_u_b
        # # v_tp_noise_b_v2 = v_tp_b_v2 + noise_v_b

        # Find params
        new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))
        new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
        new_params_b = np.array(optimiser_b(u_tp_noise_b, v_tp_noise_b, dx_b))
        # # new_params_b_v2 = np.array(optimiser_b_v2(
        # # # u_tp_noise_b_v2, v_tp_noise_b_v2, dx_b_v2))

        # # Compute relative error and add them to matrix
        # rel_error_mat_fourier_noise_sch[i, j] = np.max(
        #     abs(new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noise_fn[i, j] = np.max(
        #     abs(new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noise_b[i, j] = np.max(
        #     abs(new_params_b-c_original_b)/c_original_b)
        # # rel_error_mat_fourier_noise_b_v2[i, j] = np.max(
        #     # # # abs(new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # rel_error_mat_fourier_noise_mean_sch[i, j] = np.mean(
        #     abs(new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noise_mean_fn[i, j] = np.mean(
        #     abs(new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noise_mean_b[i, j] = np.mean(
        #     abs(new_params_b-c_original_b)/c_original_b)
        # # rel_error_mat_fourier_noise_mean_b_v2[i, j] = np.mean(
        #     # # # abs(new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # rel_error_mat_fourier_noAbs_noise_sch[i, j] = np.max(
        #     (new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noAbs_noise_fn[i, j] = np.max(
        #     (new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noAbs_noise_b[i, j] = np.max(
        #     (new_params_b-c_original_b)/c_original_b)
        # # rel_error_mat_fourier_noAbs_noise_b_v2[i, j] = np.max(
        #     # # # (new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # rel_error_mat_fourier_noAbs_noise_mean_sch[i, j] = np.mean(
        #     (new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noAbs_noise_mean_fn[i, j] = np.mean(
        #     (new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noAbs_noise_mean_b[i, j] = np.mean(
        #     (new_params_b-c_original_b)/c_original_b)
        # rel_error_mat_fourier_noAbs_noise_mean_b_v2[i, j] = np.mean(
        # # # (new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # Generate new patterns

        u_tp_sch_new, v_tp_sch_new = gen_pattern_sch(
            new_params_sch, perturbation_1, perturbation_2
        )
        u_tp_fn_new, v_tp_fn_new = gen_pattern_fn(
            new_params_fn, perturbation_1, perturbation_2
        )
        u_tp_b_new, v_tp_b_new = gen_pattern_b(
            new_params_b, perturbation_1, perturbation_2
        )
        # # # u_tp_b_v2_new, v_tp_b_v2_new = gen_pattern_b_v2(
        # new_params_b_v2, perturbation_1, perturbation_2)
        # # Now we want to compare original patterns with new patterns

        # fourier_u_sch.append(spectral_power_radial(
        #     u_tp_sch_new, u_tp_sch, plot=False))
        # fourier_v_sch.append(spectral_power_radial(
        #     v_tp_sch_new, v_tp_sch, plot=False))

        # fourier_u_fn.append(spectral_power_radial(u_tp_fn_new, u_tp_fn))
        # fourier_v_fn.append(spectral_power_radial(v_tp_fn_new, v_tp_fn))

        # fourier_u_b.append(spectral_power_radial(u_tp_b_new, u_tp_b))
        # fourier_v_b.append(spectral_power_radial(v_tp_b_new, v_tp_b))

        # # # fourier_u_b_v2.append(spectral_power_radial(u_tp_b_v2_new, u_tp_b_v2))
        # # # fourier_v_b_v2.append(spectral_power_radial(v_tp_b_v2_new, v_tp_b_v2))

        fourier_u_sch_norm.append(
            spectral_power_radial_norm(u_tp_sch_new, u_tp_sch, plot=False)
        )
        fourier_v_sch_norm.append(
            spectral_power_radial_norm(v_tp_sch_new, v_tp_sch, plot=False)
        )

        fourier_u_fn_norm.append(spectral_power_radial_norm(u_tp_fn_new, u_tp_fn))
        fourier_v_fn_norm.append(spectral_power_radial_norm(v_tp_fn_new, v_tp_fn))

        fourier_u_b_norm.append(spectral_power_radial_norm(u_tp_b_new, u_tp_b))
        fourier_v_b_norm.append(spectral_power_radial_norm(v_tp_b_new, v_tp_b))

        # fourier_u_b_v2_norm.append(
        # # spectral_power_radial_norm(u_tp_b_v2_new, u_tp_b_v2))
        # fourier_v_b_v2_norm.append(
        # # spectral_power_radial_norm(v_tp_b_v2_new, v_tp_b_v2))
        print(f"noise {i}")
    print(f"rep {j}")
# %%
n_noise = n_noise

fourier_u_sch_2 = np.reshape(fourier_u_sch, (noise_reps, n_noise))
fourier_v_sch_2 = np.reshape(fourier_v_sch, (noise_reps, n_noise))

fourier_u_fn_2 = np.reshape(fourier_u_fn, (noise_reps, n_noise))
fourier_v_fn_2 = np.reshape(fourier_v_fn, (noise_reps, n_noise))

fourier_u_b_2 = np.reshape(fourier_u_b, (noise_reps, n_noise))
fourier_v_b_2 = np.reshape(fourier_v_b, (noise_reps, n_noise))

fourier_u_b_v2_2 = np.reshape(fourier_u_b_v2, (noise_reps, n_noise))
fourier_v_b_v2_2 = np.reshape(fourier_v_b_v2, (noise_reps, n_noise))


fourier_u_sch_2_norm = np.reshape(fourier_u_sch_norm, (noise_reps, n_noise))
fourier_v_sch_2_norm = np.reshape(fourier_v_sch_norm, (noise_reps, n_noise))

fourier_u_fn_2_norm = np.reshape(fourier_u_fn_norm, (noise_reps, n_noise))
fourier_v_fn_2_norm = np.reshape(fourier_v_fn_norm, (noise_reps, n_noise))

fourier_u_b_2_norm = np.reshape(fourier_u_b_norm, (noise_reps, n_noise))
fourier_v_b_2_norm = np.reshape(fourier_v_b_norm, (noise_reps, n_noise))

fourier_u_b_v2_2_norm = np.reshape(fourier_u_b_v2_norm, (noise_reps, n_noise))
fourier_v_b_v2_2_norm = np.reshape(fourier_v_b_v2_norm, (noise_reps, n_noise))

plt.plot(noise_array, np.mean(fourier_u_sch_2, axis=0))
plt.plot(noise_array, np.mean(fourier_v_sch_2, axis=0))
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(noise_array, np.mean(fourier_u_fn_2, axis=0))
plt.plot(noise_array, np.mean(fourier_v_fn_2, axis=0))
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(noise_array, np.mean(fourier_u_b_2, axis=0))
plt.plot(noise_array, np.mean(fourier_v_b_2, axis=0))
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(noise_array, np.mean(fourier_u_b_v2_2, axis=0))
plt.plot(noise_array, np.mean(fourier_v_b_v2_2, axis=0))
plt.xscale("log")
plt.yscale("log")
plt.show()


#  Will do the scatter plot of this now
noise_points = np.tile(noise_array, (noise_reps, 1)).T.flatten()
plt.scatter(
    noise_points,
    fourier_u_sch_2[:, 30:].flatten(order="F"),
    edgecolors="none",
    s=16,
    alpha=0.4,
)
plt.scatter(
    noise_points,
    fourier_u_fn_2[:, 30:].flatten(order="F"),
    edgecolors="none",
    s=16,
    alpha=0.4,
)
plt.xscale("log")
plt.yscale("log")
mean_u_sch = np.mean(rel_error_mat_fourier_noise_mean_sch, axis=1)
plt.plot(noise_array, mean_u_sch)
std_u_sch = np.std(rel_error_mat_fourier_noise_mean_sch, axis=1)
plt.fill_between(noise_array, mean_u_sch + std_u_sch, mean_u_sch - std_u_sch, alpha=0.2)
# mean_u_fn = np.mean(rel_error_mat_fourier_noise_mean_fn, axis=1)
# plt.plot(noise_array, mean_u_fn)
# std_u_fn = np.std(rel_error_mat_fourier_noise_mean_fn, axis=1)
# plt.fill_between(noise_array, mean_u_fn+std_u_fn, mean_u_fn -
#                  std_u_fn, alpha=0.2)
plt.xscale("log")
plt.yscale("log")
x1 = noise_points
x2 = noise_points
y1 = fourier_u_sch_2[:, 30:].flatten(order="F")
y2 = fourier_u_fn_2[:, 30:].flatten(order="F")


def arr_from_fig(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


# It seems that below the threshold this gives the same values, let's see what is goinng on
imgs = list()
figsize = (4, 4)
dpi = 200

for x, y, c in zip([x1, x2], [y1, y2], ["blue", "red"]):
    fig = plt.figure(figsize=figsize, dpi=dpi, tight_layout={"pad": 0})
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=100, color=c, alpha=1)
    ax.axis("off")
    imgs.append(arr_from_fig(fig))
    plt.close()


fig = plt.figure(figsize=figsize)
alpha = 0.5

alpha_scaled = 255 * alpha
for img in imgs:
    img_alpha = np.where((img == 255).all(-1), 0, alpha_scaled).reshape(
        [*img.shape[:2], 1]
    )
    img_show = np.concatenate([img, img_alpha], axis=-1).astype(int)
    plt.imshow(img_show, origin="lower")

ticklabels = ["{:03.1f}".format(i) for i in np.linspace(-0.2, 1.2, 8, dtype=np.float16)]
plt.xticks(ticks=np.linspace(0, dpi * figsize[0], 8), labels=ticklabels)
plt.yticks(ticks=np.linspace(0, dpi * figsize[1], 8), labels=ticklabels)
# plt.xscale('log')
# plt.yscale('log')
plt.title("Test scatter")
n = 100
size = 0.02
alpha = 0.5


def points():
    x = np.random.uniform(size=n)
    y = np.random.uniform(size=n)
    return x, y


x1 = noise_points
x2 = noise_points
y1 = fourier_u_sch_2_norm[:, :].flatten(order="F")
y2 = fourier_u_fn_2_norm[:, :].flatten(order="F")
y3 = fourier_u_b_2_norm[:, :].flatten(order="F")
y4 = fourier_v_b_v2_2[:, :].flatten(order="F")
# x1, y1 = points()
# x2, y2 = points()
size = 0.02
polygons1 = [Point(x1[i], y1[i]).buffer(size) for i in range(n)]
polygons2 = [Point(x2[i], y2[i]).buffer(size) for i in range(n)]
polygons1 = cascaded_union(polygons1)
polygons2 = cascaded_union(polygons2)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, title="Test scatter")
for polygon1 in polygons1:
    polygon1 = ptc.Polygon(np.array(polygon1.exterior), lw=0, alpha=alpha)
    ax.add_patch(polygon1)
for polygon2 in polygons2:
    polygon2 = ptc.Polygon(np.array(polygon2.exterior), lw=0, alpha=alpha)
    ax.add_patch(polygon2)
ax.axis([-0.2, 1.2, -0.2, 1.2])

############ PAPER FIGURE ################
n_noise1 = n_noise
n_noise2 = 25

# We need to split the arrays into two

fourier_u_sch_p1 = fourier_u_sch[: noise_reps * n_noise]
fourier_u_sch_p2 = fourier_u_sch[noise_reps * n_noise :]

fourier_u_sch_2 = np.reshape(fourier_u_sch_p1, (noise_reps, n_noise1))
fourier_u_sch_2 = np.hstack(
    [fourier_u_sch_2, np.reshape(fourier_u_sch_p2, (noise_reps, n_noise2))]
)

fourier_v_sch_p1 = fourier_v_sch[: noise_reps * n_noise]
fourier_v_sch_p2 = fourier_v_sch[noise_reps * n_noise :]

fourier_v_sch_2 = np.reshape(fourier_v_sch_p1, (noise_reps, n_noise1))
fourier_v_sch_2 = np.hstack(
    [fourier_v_sch_2, np.reshape(fourier_v_sch_p2, (noise_reps, n_noise2))]
)

fourier_u_fn_p1 = fourier_u_fn[: noise_reps * n_noise]
fourier_u_fn_p2 = fourier_u_fn[noise_reps * n_noise :]

fourier_u_fn_2 = np.reshape(fourier_u_fn_p1, (noise_reps, n_noise1))
fourier_u_fn_2 = np.hstack(
    [fourier_u_fn_2, np.reshape(fourier_u_fn_p2, (noise_reps, n_noise2))]
)

fourier_v_fn_p1 = fourier_v_fn[: noise_reps * n_noise]
fourier_v_fn_p2 = fourier_v_fn[noise_reps * n_noise :]

fourier_v_fn_2 = np.reshape(fourier_v_fn_p1, (noise_reps, n_noise1))
fourier_v_fn_2 = np.hstack(
    [fourier_v_fn_2, np.reshape(fourier_v_fn_p2, (noise_reps, n_noise2))]
)

fourier_u_b_p1 = fourier_u_b[: noise_reps * n_noise]
fourier_u_b_p2 = fourier_u_b[noise_reps * n_noise :]

fourier_u_b_2 = np.reshape(fourier_u_b_p1, (noise_reps, n_noise1))
fourier_u_b_2 = np.hstack(
    [fourier_u_b_2, np.reshape(fourier_u_b_p2, (noise_reps, n_noise2))]
)

fourier_v_b_p1 = fourier_v_b[: noise_reps * n_noise]
fourier_v_b_p2 = fourier_v_b[noise_reps * n_noise :]

fourier_v_b_2 = np.reshape(fourier_v_b_p1, (noise_reps, n_noise1))
fourier_v_b_2 = np.hstack(
    [fourier_v_b_2, np.reshape(fourier_v_b_p2, (noise_reps, n_noise2))]
)


fourier_u_sch_p1 = fourier_u_sch_norm[: noise_reps * n_noise]
fourier_u_sch_p2 = fourier_u_sch_norm[noise_reps * n_noise :]

fourier_u_sch_2_norm = np.reshape(fourier_u_sch_p1, (noise_reps, n_noise1))
fourier_u_sch_2_norm = np.hstack(
    [fourier_u_sch_2_norm, np.reshape(fourier_u_sch_p2, (noise_reps, n_noise2))]
)

fourier_v_sch_p1 = fourier_v_sch_norm[: noise_reps * n_noise]
fourier_v_sch_p2 = fourier_v_sch_norm[noise_reps * n_noise :]

fourier_v_sch_2_norm = np.reshape(fourier_v_sch_p1, (noise_reps, n_noise1))
fourier_v_sch_2_norm = np.hstack(
    [fourier_v_sch_2_norm, np.reshape(fourier_v_sch_p2, (noise_reps, n_noise2))]
)

fourier_u_fn_p1 = fourier_u_fn_norm[: noise_reps * n_noise]
fourier_u_fn_p2 = fourier_u_fn_norm[noise_reps * n_noise :]

fourier_u_fn_2_norm = np.reshape(fourier_u_fn_p1, (noise_reps, n_noise1))
fourier_u_fn_2_norm = np.hstack(
    [fourier_u_fn_2_norm, np.reshape(fourier_u_fn_p2, (noise_reps, n_noise2))]
)

fourier_v_fn_p1 = fourier_v_fn_norm[: noise_reps * n_noise]
fourier_v_fn_p2 = fourier_v_fn_norm[noise_reps * n_noise :]

fourier_v_fn_2_norm = np.reshape(fourier_v_fn_p1, (noise_reps, n_noise1))
fourier_v_fn_2_norm = np.hstack(
    [fourier_v_fn_2_norm, np.reshape(fourier_v_fn_p2, (noise_reps, n_noise2))]
)

fourier_u_b_p1 = fourier_u_b_norm[: noise_reps * n_noise]
fourier_u_b_p2 = fourier_u_b_norm[noise_reps * n_noise :]

fourier_u_b_2_norm = np.reshape(fourier_u_b_p1, (noise_reps, n_noise1))
fourier_u_b_2_norm = np.hstack(
    [fourier_u_b_2_norm, np.reshape(fourier_u_b_p2, (noise_reps, n_noise2))]
)

fourier_v_b_p1 = fourier_v_b_norm[: noise_reps * n_noise]
fourier_v_b_p2 = fourier_v_b_norm[noise_reps * n_noise :]

fourier_v_b_2_norm = np.reshape(fourier_v_b_p1, (noise_reps, n_noise1))
fourier_v_b_2_norm = np.hstack(
    [fourier_v_b_2, np.reshape(fourier_v_b_p2, (noise_reps, n_noise2))]
)


noise_array_new = np.hstack([noise_array, noise_array_2])


def expand(x, y, gap=1e-7):
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    return x1, y1


np.shape(fourier_u_sch_norm)

skip_n = 0
noise_points = np.tile(noise_array[skip_n:], (noise_reps, 1)).flatten()
x1 = noise_points
x2 = noise_points
# y1 = fourier_u_sch_2[:, skip_n:].flatten(order='F')
# y2 = fourier_u_fn_2[:, skip_n:].flatten(order='F')
# y3 = fourier_u_b_2[:, skip_n:].flatten(order='F')
# y4 = fourier_u_b_v2_2[:, skip_n:].flatten(order='F')
# y1_norm = fourier_u_sch_norm[:, skip_n:].flatten(order='F')
# y2_norm = fourier_u_fn_norm[:, skip_n:].flatten(order='F')
# y3_norm = fourier_u_b_norm[:, skip_n:].flatten(order='F')
y1_norm = fourier_u_sch_norm
y2_norm = fourier_u_fn_norm
y3_norm = fourier_u_b_norm
# y4_norm = fourier_v_b_v2_2_norm[:, skip_n:].flatten(order='F')

plt.rcParams["lines.solid_capstyle"] = "round"

# x1, y1 = points()
# x2, y2 = points()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
# ax.plot(*expand(x1, y1), lw=5, alpha=0.3)
# ax.plot(*expand(x2, y2), lw=5, alpha=0.3)
# ax.plot(*expand(x2, y3), lw=5, alpha=0.6)
# ax.plot(*expand(x2, y4_norm), lw=5, alpha=0.6)
# ax.plot(*expand(x1, y1_norm), color='tab:blue', lw=5, alpha=0.3)
# ax.plot(*expand(x2, y2_norm), color='tab:orange', lw=5, alpha=0.3)
# ax.plot(*expand(x2, y3_norm), color='tab:green', lw=5, alpha=0.6)
ax.plot(*expand(x1, y1_norm), color="tab:blue", lw=5, alpha=0.6)
ax.plot(*expand(x2, y2_norm), color="tab:orange", lw=5, alpha=0.6)
# ax.plot(*expand(x2, y4), lw=5, alpha=0.6)
plt.xscale("log")
plt.yscale("log")
ax.set_xlabel("Relative Noise (%)", fontsize=25)
ax.set_ylabel("RAPSD", fontsize=25)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.tick_params(axis="both", which="minor", labelsize=20)
# ax.set_xlim([0.04,4])
# ax.set_ylim([1e-3,1.1])
# ,'Brusselator'], fontsize=14)
plt.legend(["Schnakenberg", "FitzHugh-Nagumo"], fontsize=20)
fig.savefig("Scatter_RAPS_labels_legend_S_FN_VF.pdf")
fig.savefig("Scatter_RAPS_labels_legend_S_FN_VF")
fig.savefig("Scatter_RAPS_labels_no_legend_all_VF.pdf")
fig.savefig("Scatter_RAPS_labels_no_legend_all_VF")
np.save("Scatter_plot_RAPSD", [x1, y1_norm, y2_norm, y3_norm])
###############               ####################
# %% Run the noise plots
n_noise = 30
noise_reps = 10
noise_array = np.geomspace(1e-3, 1, n_noise)
# rel_error_mat_fourier_noise_sch_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_fn_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_b_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_b_v2_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_mean_sch_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_mean_fn_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_mean_b_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noise_mean_b_v2_3 = np.zeros((n_noise, noise_reps))

# rel_error_mat_fourier_noAbs_noise_sch_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_fn_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_b_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_b_v2_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_mean_sch_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_mean_fn_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_mean_b_3 = np.zeros((n_noise, noise_reps))
# rel_error_mat_fourier_noAbs_noise_mean_b_v2_3 = np.zeros((n_noise, noise_reps))

params_sch = []
params_fn = []
params_b = []
fourier_u_sch_3 = np.zeros((n_noise, noise_reps))
fourier_v_sch_3 = np.zeros((n_noise, noise_reps))
fourier_u_fn_3 = np.zeros((n_noise, noise_reps))
fourier_v_fn_3 = np.zeros((n_noise, noise_reps))
fourier_u_b_3 = np.zeros((n_noise, noise_reps))
fourier_v_b_3 = np.zeros((n_noise, noise_reps))
fourier_u_b_v2_3 = np.zeros((n_noise, noise_reps))
fourier_v_b_v2_3 = np.zeros((n_noise, noise_reps))

fourier_u_sch_3_norm = np.zeros((n_noise, noise_reps))
fourier_v_sch_3_norm = np.zeros((n_noise, noise_reps))
fourier_u_fn_3_norm = np.zeros((n_noise, noise_reps))
fourier_v_fn_3_norm = np.zeros((n_noise, noise_reps))
fourier_u_b_3_norm = np.zeros((n_noise, noise_reps))
fourier_v_b_3_norm = np.zeros((n_noise, noise_reps))
fourier_u_b_v2_3_norm = np.zeros((n_noise, noise_reps))
fourier_v_b_v2_3_norm = np.zeros((n_noise, noise_reps))

for j in range(noise_reps):
    noise_lev = 1
    for i, noise_lev in enumerate(noise_array):
        # Add noise to patterns

        # Schnakenberg
        noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
        noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
        noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
        noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
        u_tp_noise_sch = u_tp_sch + noise_u_sch
        v_tp_noise_sch = v_tp_sch + noise_v_sch

        # FitzHugh-Nagumo
        noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
        noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
        noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
        noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
        u_tp_noise_fn = u_tp_fn + noise_u_fn
        v_tp_noise_fn = v_tp_fn + noise_v_fn

        # Brusselator
        noise_spread_u_b = noise_lev / 100 * (np.max(u_tp_b) - np.min(u_tp_b))
        noise_spread_v_b = noise_lev / 100 * (np.max(v_tp_b) - np.min(v_tp_b))
        noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
        noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
        u_tp_noise_b = u_tp_b + noise_u_b
        v_tp_noise_b = v_tp_b + noise_v_b

        # Brusselator v2
        noise_spread_u_b = noise_lev / 100 * (np.max(u_tp_b_v2) - np.min(u_tp_b_v2))
        noise_spread_v_b = noise_lev / 100 * (np.max(v_tp_b_v2) - np.min(v_tp_b_v2))
        noise_u_b = np.random.normal(0, noise_spread_u_b, (n, n))
        noise_v_b = np.random.normal(0, noise_spread_v_b, (n, n))
        u_tp_noise_b_v2 = u_tp_b_v2 + noise_u_b
        v_tp_noise_b_v2 = v_tp_b_v2 + noise_v_b

        # Find params
        new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))
        new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
        new_params_b = np.array(optimiser_b(u_tp_noise_b, v_tp_noise_b, dx_b))
        new_params_b_v2 = np.array(
            optimiser_b_v2(u_tp_noise_b_v2, v_tp_noise_b_v2, dx_b_v2)
        )

        # Find params
        new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))
        new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))
        u_tp_sch_1, v_tp_sch_1 = gen_pattern_sch(
            new_params_sch, perturbation_1, perturbation_2
        )
        u_tp_fn_1, v_tp_fn_1 = gen_pattern_fn(
            new_params_fn, perturbation_1, perturbation_2
        )
        plt.imshow(u_tp_sch_1)
        plt.imshow(u_tp_fn)
        ##########     FIGURE FOR PAPER      ##################

        tp_new_1 = (u_tp_sch_1 - np.min(u_tp_sch_1)) / (
            np.max(u_tp_sch_1) - np.min(u_tp_sch_1)
        )
        tp_new_9em1 = (u_tp_sch_9em1 - np.min(u_tp_sch_9em1)) / (
            np.max(u_tp_sch_9em1) - np.min(u_tp_sch_9em1)
        )
        tp_new_8em1 = (u_tp_sch_8em1 - np.min(u_tp_sch_8em1)) / (
            np.max(u_tp_sch_8em1) - np.min(u_tp_sch_8em1)
        )
        tp_new_6em1 = (u_tp_sch_6em1 - np.min(u_tp_sch_6em1)) / (
            np.max(u_tp_sch_6em1) - np.min(u_tp_sch_6em1)
        )
        tp_new_3em1 = (u_tp_sch_3em1 - np.min(u_tp_sch_3em1)) / (
            np.max(u_tp_sch_3em1) - np.min(u_tp_sch_3em1)
        )
        tp_new_1em1 = (u_tp_sch_1em1 - np.min(u_tp_sch_1em1)) / (
            np.max(u_tp_sch_1em1) - np.min(u_tp_sch_1em1)
        )
        tp_or_sch = (u_tp_sch - np.min(u_tp_sch)) / (
            np.max(u_tp_sch) - np.min(u_tp_sch)
        )
        tp_sp_p_1 = np.abs(np.fft.fftshift(np.fft.fft2(tp_or_sch)[1:, 1:])) ** 2
        tp_sp_p_2 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new_1em1)[1:, 1:])) ** 2
        tp_sp_p_3 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new_3em1)[1:, 1:])) ** 2
        tp_sp_p_4 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new_6em1)[1:, 1:])) ** 2
        tp_sp_p_5 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new_8em1)[1:, 1:])) ** 2
        tp_sp_p_6 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new_9em1)[1:, 1:])) ** 2
        tp_sp_p_7 = np.abs(np.fft.fftshift(np.fft.fft2(tp_new_1)[1:, 1:])) ** 2

        x, y = np.meshgrid(np.arange(tp_sp_p_1.shape[1]), np.arange(tp_sp_p_1.shape[0]))
        R = np.sqrt(x**2 + y**2)

        def f1(r):
            return tp_sp_p_1[(R >= r - 2) & (R < r + 2)].mean()

        r1 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
        mean1 = np.vectorize(f1)(r1)

        def f2(r):
            return tp_sp_p_2[(R >= r - 2) & (R < r + 2)].mean()

        r2 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
        mean2 = np.vectorize(f2)(r2)

        def f3(r):
            return tp_sp_p_3[(R >= r - 2) & (R < r + 2)].mean()

        r3 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
        mean3 = np.vectorize(f3)(r3)

        def f4(r):
            return tp_sp_p_4[(R >= r - 2) & (R < r + 2)].mean()

        r4 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
        mean4 = np.vectorize(f4)(r4)

        def f5(r):
            return tp_sp_p_5[(R >= r - 2) & (R < r + 2)].mean()

        r5 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
        mean5 = np.vectorize(f5)(r5)

        def f6(r):
            return tp_sp_p_6[(R >= r - 2) & (R < r + 2)].mean()

        r6 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
        mean6 = np.vectorize(f6)(r6)

        def f7(r):
            return tp_sp_p_7[(R >= r - 2) & (R < r + 2)].mean()

        r7 = np.linspace(3, int(np.max(R)), num=int(np.max(R)) - 2)
        mean7 = np.vectorize(f7)(r7)

        fig, ax = plt.subplots()
        ax.plot(r1, mean1)
        # ax.plot(r2, mean2)
        ax.plot(r3, mean3)
        ax.plot(r4, mean4)
        # ax.plot(r5, mean5)
        # ax.plot(r6, mean6)
        ax.plot(r7, mean7)
        # plt.yscale('log')
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)

        plt.savefig("Fourier_spectral_power_sch")
        plt.savefig("Fourier_spectral_power_sch.pdf")
        plt.show()

        ###############               ####################
        # Compute relative error and add them to matrix
        # rel_error_mat_fourier_noise_sch_3[i, j] = np.max(
        #     abs(new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noise_fn_3[i, j] = np.max(
        #     abs(new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noise_b_3[i, j] = np.max(
        #     abs(new_params_b-c_original_b)/c_original_b)
        # rel_error_mat_fourier_noise_b_v2_3[i, j] = np.max(
        #     abs(new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # rel_error_mat_fourier_noise_mean_sch_3[i, j] = np.mean(
        #     abs(new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noise_mean_fn_3[i, j] = np.mean(
        #     abs(new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noise_mean_b_3[i, j] = np.mean(
        #     abs(new_params_b-c_original_b)/c_original_b)
        # rel_error_mat_fourier_noise_mean_b_v2_3[i, j] = np.mean(
        #     abs(new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # rel_error_mat_fourier_noAbs_noise_sch_3[i, j] = np.max(
        #     (new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noAbs_noise_fn_3[i, j] = np.max(
        #     (new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noAbs_noise_b_3[i, j] = np.max(
        #     (new_params_b-c_original_b)/c_original_b)
        # rel_error_mat_fourier_noAbs_noise_b_v2_3[i, j] = np.max(
        #     (new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # rel_error_mat_fourier_noAbs_noise_mean_sch_3[i, j] = np.mean(
        #     (new_params_sch-c_original_sch)/c_original_sch)
        # rel_error_mat_fourier_noAbs_noise_mean_fn_3[i, j] = np.mean(
        #     (new_params_fn-c_original_fn)/c_original_fn)
        # rel_error_mat_fourier_noAbs_noise_mean_b_3[i, j] = np.mean(
        #     (new_params_b-c_original_b)/c_original_b)
        # rel_error_mat_fourier_noAbs_noise_mean_b_v2_3[i, j] = np.mean(
        #     (new_params_b_v2-c_original_b_v2)/c_original_b_v2)

        # Generate new patterns

        u_tp_sch_new, v_tp_sch_new = gen_pattern_sch(
            new_params_sch, perturbation_1, perturbation_2
        )
        u_tp_fn_new, v_tp_fn_new = gen_pattern_fn(
            new_params_fn, perturbation_1, perturbation_2
        )
        u_tp_b_new, v_tp_b_new = gen_pattern_b(
            new_params_b, perturbation_1, perturbation_2
        )
        u_tp_b_v2_new, v_tp_b_v2_new = gen_pattern_b_v2(
            new_params_b_v2, perturbation_1, perturbation_2
        )
        plt.imshow(u_tp_fn_new, cmap=cmap)
        plt.colorbar()
        plt.axis("off")
        plt.show()
        plt.imshow(u_tp_fn, cmap=cmap)
        plt.colorbar()
        plt.axis("off")
        plt.show()
        plt.imshow(u_tp_sch_new, cmap=cmap)
        plt.colorbar()
        plt.axis("off")
        plt.show()
        plt.imshow(u_tp_sch, cmap=cmap)
        plt.colorbar()
        plt.axis("off")
        plt.show()

        plt.imshow(u_tp_fn_new, cmap=cmap)
        plt.axis("off")
        plt.savefig("Pattern_FN_1.pdf")
        plt.savefig("Pattern_FN_1")
        plt.imshow(u_tp_sch_new, cmap=cmap)
        # plt.colorbar()
        plt.axis("off")
        plt.savefig("Pattern_sch_1.pdf")
        plt.savefig("Pattern_sch_1")

        # plt.imshow(v_tp_b_new, cmap=cmap)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(v_tp_b, cmap=cmap)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(v_tp_sch_new, cmap=cmap)
        # plt.colorbar()
        # plt.imshow(v_tp_sch, cmap=cmap)
        # plt.colorbar()

        # Now we want to compare original patterns with new patterns
        fourier_u_sch_3[i, j] = spectral_power_radial(u_tp_sch_new, u_tp_sch, plot=True)
        fourier_v_sch_3[i, j] = spectral_power_radial(v_tp_sch_new, v_tp_sch)
        fourier_u_fn_3[i, j] = spectral_power_radial(u_tp_fn_new, u_tp_fn)
        fourier_v_fn_3[i, j] = spectral_power_radial(v_tp_fn_new, v_tp_fn)
        fourier_u_b_3[i, j] = spectral_power_radial(u_tp_b_new, u_tp_b)
        fourier_v_b_3[i, j] = spectral_power_radial(v_tp_b_new, v_tp_b)
        fourier_u_b_v2_3[i, j] = spectral_power_radial(u_tp_b_v2_new, u_tp_b_v2)
        fourier_v_b_v2_3[i, j] = spectral_power_radial(v_tp_b_v2_new, v_tp_b_v2)
        #  New measure
        fourier_u_sch_3_norm[i, j] = spectral_power_radial_norm(u_tp_sch_new, u_tp_sch)
        fourier_v_sch_3_norm[i, j] = spectral_power_radial_norm(v_tp_sch_new, v_tp_sch)
        fourier_u_fn_3_norm[i, j] = spectral_power_radial_norm(u_tp_fn_new, u_tp_fn)
        fourier_v_fn_3_norm[i, j] = spectral_power_radial_norm(v_tp_fn_new, v_tp_fn)
        fourier_u_b_3_norm[i, j] = spectral_power_radial_norm(u_tp_b_new, u_tp_b)
        fourier_v_b_3_norm[i, j] = spectral_power_radial_norm(v_tp_b_new, v_tp_b)
        fourier_u_b_v2_3_norm[i, j] = spectral_power_radial_norm(
            u_tp_b_v2_new, u_tp_b_v2
        )
        fourier_v_b_v2_3_norm[i, j] = spectral_power_radial_norm(
            v_tp_b_v2_new, v_tp_b_v2
        )
        # #  New measure_2
        # fourier_u_sch_3_norm[i, j] = spectral_power_radial_norm_v2(
        #     u_tp_sch_new, u_tp_sch)
        # fourier_v_sch_3_norm[i, j] = spectral_power_radial_norm_v2(
        #     v_tp_sch_new, v_tp_sch)
        # fourier_u_fn_3_norm[i, j] = spectral_power_radial_norm_v2(u_tp_fn_new, u_tp_fn)
        # fourier_v_fn_3_norm[i, j] = spectral_power_radial_norm_v2(v_tp_fn_new, v_tp_fn)
        # fourier_u_b_3_norm[i, j] = spectral_power_radial_norm_v2(u_tp_b_new, u_tp_b)
        # fourier_v_b_3_norm[i, j] = spectral_power_radial_norm_v2(v_tp_b_new, v_tp_b)
        # fourier_u_b_v2_3_norm[i, j] = spectral_power_radial_norm_v2(
        #     u_tp_b_v2_new, u_tp_b_v2)
        # fourier_v_b_v2_3_norm[i, j] = spectral_power_radial_norm_v2(
        #     v_tp_b_v2_new, v_tp_b_v2)


spectral_power_radial(u_tp_fn_new, u_tp_fn, plot=True)
spectral_power_radial(u_tp_sch_new, u_tp_sch, plot=True)
spectral_power_radial(u_tp_b_new, u_tp_b, plot=True)
# DO A SCATTER PLOT
noise_points = np.tile(noise_array, (noise_reps, 1)).T.flatten()
plt.scatter(noise_points, fourier_u_sch_3_norm.flatten())
plt.xscale("log")
plt.yscale("log")
mean_u_sch = np.mean(fourier_u_sch_3_norm, axis=1)
# mean_v_sch = np.mean(fourier_v_sch_3_norm, axis=1)
std_u_sch = np.std(fourier_u_sch_3_norm, axis=1)
std_v_sch = np.std(fourier_v_sch_3_norm, axis=1)
plt.plot(noise_array, mean_u_sch)
# plt.plot(noise_array, mean_v_sch)
plt.fill_between(
    noise_array,
    mean_u_sch + std_u_sch,
    mean_u_sch - std_u_sch,
    color="purple",
    alpha=0.2,
)
# plt.fill_between(noise_array, mean_v_sch+std_v_sch, mean_v_sch -
#  std_v_sch, color='purple', alpha=0.2)
plt.title("Schnakenberg")
plt.xlabel("Relative noise")
plt.ylabel("Difference in Radial Power")
plt.xscale("log")
plt.yscale("log")
# plt.show()

mean_u_fn = np.mean(fourier_u_fn_3_norm, axis=1)
# mean_v_fn = np.mean(fourier_v_fn_3_norm, axis=1)
std_u_fn = np.std(fourier_u_fn_3_norm, axis=1)
std_v_fn = np.std(fourier_v_fn_3_norm, axis=1)
plt.plot(noise_array, mean_u_fn)
# plt.plot(noise_array, mean_v_fn)
plt.fill_between(
    noise_array, mean_u_fn + std_u_fn, mean_u_fn - std_u_fn, color="purple", alpha=0.2
)
# plt.fill_between(noise_array, mean_v_fn+std_v_fn, mean_v_fn -
#  std_v_fn, color='purple', alpha=0.2)

plt.xscale("log")
plt.yscale("log")
plt.title("FitzHugh-Nagumo")
plt.xlabel("Relative noise")
plt.ylabel("Difference in Radial Power")
plt.show()

plt.scatter(noise_points, fourier_u_fn_3_norm.flatten())
plt.xscale("log")
plt.yscale("log")

mean_u_b = np.mean(fourier_u_b_3_norm, axis=1)
mean_v_b = np.mean(fourier_v_b_3_norm, axis=1)
std_u_b = np.std(fourier_u_b_3_norm, axis=1)
std_v_b = np.std(fourier_v_b_3_norm, axis=1)
plt.plot(noise_array, mean_u_b)
plt.plot(noise_array, mean_v_b)
plt.fill_between(
    noise_array, mean_u_b + std_u_b, mean_u_b - std_u_b, color="purple", alpha=0.2
)
plt.fill_between(
    noise_array, mean_v_b + std_v_b, mean_v_b - std_v_b, color="purple", alpha=0.2
)
plt.title("Brusselator")
plt.xlabel("Relative noise")
plt.ylabel("Difference in Radial Power")

plt.xscale("log")
plt.yscale("log")
plt.show()

plt.scatter(noise_points, fourier_u_b_3_norm.flatten())
plt.xscale("log")
plt.yscale("log")

mean_u_b_v2 = np.mean(fourier_u_b_v2_3_norm, axis=1)
mean_v_b_v2 = np.mean(fourier_v_b_v2_3_norm, axis=1)
std_u_b_v2 = np.std(fourier_u_b_v2_3_norm, axis=1)
std_v_b_v2 = np.std(fourier_v_b_v2_3_norm, axis=1)
plt.plot(noise_array, mean_u_b_v2)
plt.plot(noise_array, mean_v_b_v2)
plt.fill_between(
    noise_array,
    mean_u_b_v2 + std_u_b_v2,
    mean_u_b_v2 - std_u_b_v2,
    color="purple",
    alpha=0.2,
)
plt.fill_between(
    noise_array,
    mean_v_b_v2 + std_v_b_v2,
    mean_v_b_v2 - std_v_b_v2,
    color="purple",
    alpha=0.2,
)
plt.title("Brusselator_v2")
plt.xlabel("Relative noise")
plt.ylabel("Difference in Radial Power")

plt.xscale("log")
plt.yscale("log")
plt.show()

plt.scatter(noise_points, fourier_u_b_v2_3_norm.flatten())
plt.xscale("log")
plt.yscale("log")

# Scatter plots of sch and fn

plt.scatter(noise_points, fourier_u_sch_3_norm.flatten())
plt.scatter(noise_points, fourier_u_fn_3_norm.flatten())
# plt.xscale('log')
# plt.yscale('log')
plt.legend(["Schnakenberg", "FitzHugh-Nagumo"])
plt.savefig("Fourier_Radial_Spectrum_FN_S")

plt.plot(noise_array, np.mean(fourier_u_fn_2, axis=1))
plt.plot(noise_array, np.mean(fourier_v_fn_2, axis=1))
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(noise_array, np.mean(fourier_u_b_2, axis=1))
plt.plot(noise_array, np.mean(fourier_v_b_2, axis=1))
plt.xscale("log")
plt.yscale("log")
plt.show()

plt.plot(noise_array, np.mean(fourier_u_b_v2_2, axis=1))
plt.plot(noise_array, np.mean(fourier_v_b_v2_2, axis=1))
plt.xscale("log")
plt.yscale("log")
plt.show()


# Make pattern plots
def gen_pattern_sch_noise(noise_lev):
    noise_spread_u_sch = noise_lev / 100 * (np.max(u_tp_sch) - np.min(u_tp_sch))
    noise_spread_v_sch = noise_lev / 100 * (np.max(v_tp_sch) - np.min(v_tp_sch))
    noise_u_sch = np.random.normal(0, noise_spread_u_sch, (n, n))
    noise_v_sch = np.random.normal(0, noise_spread_v_sch, (n, n))
    u_tp_noise_sch = u_tp_sch + noise_u_sch
    v_tp_noise_sch = v_tp_sch + noise_v_sch

    # Find params
    new_params_sch = np.array(optimiser_sch(u_tp_noise_sch, v_tp_noise_sch, dx_sch))

    u_tp_sch_1, v_tp_sch_1 = gen_pattern_sch(
        new_params_sch, perturbation_1, perturbation_2
    )
    return (u_tp_sch_1, v_tp_sch_1)


def gen_pattern_fn_noise(noise_lev):
    noise_spread_u_fn = noise_lev / 100 * (np.max(u_tp_fn) - np.min(u_tp_fn))
    noise_spread_v_fn = noise_lev / 100 * (np.max(v_tp_fn) - np.min(v_tp_fn))
    noise_u_fn = np.random.normal(0, noise_spread_u_fn, (n, n))
    noise_v_fn = np.random.normal(0, noise_spread_v_fn, (n, n))
    u_tp_noise_fn = u_tp_fn + noise_u_fn
    v_tp_noise_fn = v_tp_fn + noise_v_fn

    # Find params
    new_params_fn = np.array(optimiser_fn(u_tp_noise_fn, v_tp_noise_fn, dx_fn))

    u_tp_fn_1, v_tp_fn_1 = gen_pattern_fn(new_params_fn, perturbation_1, perturbation_2)
    return (u_tp_fn_1, v_tp_fn_1)


noise_levels = [2e-2, 2e-1, 5e-1, 9e-1, 15e-1]
for noise in noise_levels:
    u_sch, v_sch = gen_pattern_sch_noise(noise)
    u_fn, v_fn = gen_pattern_fn_noise(noise)
    plt.imshow(u_sch, cmap=cmap)
    plt.axis("off")
    plt.savefig(f"Pattern_sch_{noise:.2e}".replace(".", "_") + "_v2.pdf")
    plt.savefig(f"Pattern_sch_{noise:.2e}_v2".replace(".", "_"))
    plt.imshow(u_fn, cmap=cmap)
    plt.axis("off")
    plt.savefig(f"Pattern_fn_{noise:.2e}".replace(".", "_") + "_v2.pdf")
    plt.savefig(f"Pattern_fn_{noise:.2e}_v2".replace(".", "_"))
