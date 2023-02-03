# %% [markdown]
# Use the Euler method to solve the ODE $\dot{x} = x$ with initial condition $x(0) = 1$.
# You should use this to estimate $x(1)$ using different step sizes.

# Also make a function called ***solve_to*** which solves from $x_1$, $t_1$ to $x_2$, $t_2$ in steps no bigger than **deltat_max**. This should be similar to scipy's ***odeint*** function.

# %%
import numpy as np
import matplotlib.pyplot as plt
import timeit

# %%
# Euler method


def euler_step(f, u, t, dt):
    """Compute the solution at the next time-step using Euler's method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation du/dt = f(u, t).
    u : float
        Solution at the previous time-step.
    t : float
        The current time.
    dt : float
        Time-step size.

    Returns
    -------
    u_n_plus_1 : float
        Approximate solution at the next time-step.
    """
    u_n_plus_1 = u + dt*f(u, t)
    return u_n_plus_1

# %%
# solve to


def solve_to(f, u0, t0, t1, dt, method=euler_step):
    """Solve the ODE du/dt = f(u, t) from t0 to t1 with initial condition u0.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation du/dt = f(u, t).
    u0 : float
        Initial condition.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    dt : float
        Time-step size.
    method : function, optional
        Numerical method to use to solve the ODE.

    Returns
    -------
    u : numpy.ndarray
        Array of approximate solutions at each time-step.
    t : numpy.ndarray
        Array of time points corresponding to the time-steps.
    """
    # Number of time-steps.
    N = int(np.ceil((t1 - t0)/dt))

    # Initialise the arrays to store the solution and time.
    u = np.zeros(N + 1)
    t = np.zeros(N + 1)

    # Set the initial condition.
    u[0] = u0
    t[0] = t0

    # Time loop.
    for n in range(N):
        t[n + 1] = t[n] + dt
        u[n + 1] = euler_step(f, u[n], t[n], dt)

    return u, t

# %%
# Solve the ODE using the Euler method.


x0 = 1.0
t0 = 0.0
t1 = 1.0
deltat_max = 0.1
def func(x, t): return x


u, t = solve_to(func, x0, t0, t1, deltat_max)
print(u[-1])

# %% [markdown]

# Produce a(nicely formatted) plot with double logarithmic scale showing how the error depends on the size of the timestep $\Delta t$.

# %%
# Plot the error as a function of the time-step size.

timesteps = np.logspace(-2, 0, 100)
errors = np.zeros_like(timesteps)
for i, dt in enumerate(timesteps):
    u, t = solve_to(func, x0, t0, t1, dt)
    errors[i] = np.abs(u[-1] - np.exp(t1))

plt.loglog(timesteps, errors, 'o')
plt.xlabel(r'$\Delta t$')
plt.ylabel('Error')
plt.show()

# %% [markdown]
# Repeat part 1 using the 4th-order Runge-Kutta method.

# %%
# Runge-Kutta method


def rk4_step(f, u, t, dt):
    """Compute the solution at the next time-step using the 4th-order
    Runge-Kutta method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation du/dt = f(u, t).
    u : float
        Solution at the previous time-step.
    t : float
        The current time.
    dt : float
        Time-step size.

    Returns
    -------
    u_n_plus_1 : float
        Approximate solution at the next time-step.
    """
    k1 = dt*f(u, t)
    k2 = dt*f(u + 0.5*k1, t + 0.5*dt)
    k3 = dt*f(u + 0.5*k2, t + 0.5*dt)
    k4 = dt*f(u + k3, t + dt)
    u_n_plus_1 = u + (k1 + 2*k2 + 2*k3 + k4)/6
    return u_n_plus_1

# %%
# Solve the ODE using the Euler method.


x0 = 1.0
t0 = 0.0
t1 = 1.0
deltat_max = 0.1
def func(x, t): return x


u, t = solve_to(func, x0, t0, t1, deltat_max, method=rk4_step)
print(u[-1])

# %% [markdown]
# Produce a(nicely formatted) plot with double logarithmic scale showing how the error depends on the size of the timestep $\Delta t$.

# %%
# Plot the error as a function of the time-step size.

timesteps = np.logspace(-2, 0, 100)
errors = np.zeros_like(timesteps)
for i, dt in enumerate(timesteps):
    u, t = solve_to(func, x0, t0, t1, dt, method=rk4_step)
    errors[i] = np.abs(u[-1] - np.exp(t1))

plt.loglog(timesteps, errors, 'o')
plt.xlabel(r'$\Delta t$')
plt.ylabel('Error')
plt.show()

# %% [markdown]
# Find step-sizes for each method that give you the same error - how long does each method take? (you can use the time command when running your Python script)

# %%
# Time of execution for equivalent error for each method.
# prettier-ignore

# Note that due to prettier formatting (on save), the code below has to be uncommented before it can be run.


def time_methods():
    print('Euler method')
    # a = %timeit -o solve_to(func, x0, t0, t1, deltat_max, method=euler_step)
    print('RK4 method')
    # b = %timeit -o solve_to(func, x0, t0, t1, deltat_max, method=rk4_step)


time_methods()
# %% [markdown]
# Extend your Euler and RK4 routines to be able to work with systems of ODEs.

# %%
# System of ODEs


def euler_step_system(f, u, t, dt):
    """Compute the solution at the next time-step using Euler's method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation du/dt = f(u, t).
    u : numpy.ndarray
        Solution at the previous time-step.
    t : float
        The current time.
    dt : float
        Time-step size.

    Returns
    -------
    u_n_plus_1 : numpy.ndarray
        Approximate solution at the next time-step.
    """
    u_n_plus_1 = u + dt*f(u, t)
    return u_n_plus_1


def rk4_step_system(f, u, t, dt):
    """Compute the solution at the next time-step using the 4th-order
    Runge-Kutta method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation du/dt = f(u, t).
    u : numpy.ndarray
        Solution at the previous time-step.
    t : float
        The current time.
    dt : float
        Time-step size.

    Returns
    -------
    u_n_plus_1 : numpy.ndarray
        Approximate solution at the next time-step.
    """
    k1 = dt*f(u, t)
    k2 = dt*f(u + 0.5*k1, t + 0.5*dt)
    k3 = dt*f(u + 0.5*k2, t + 0.5*dt)
    k4 = dt*f(u + k3, t + dt)
    u_n_plus_1 = u + (k1 + 2*k2 + 2*k3 + k4)/6
    return u_n_plus_1


def solve_to_system(f, u0, t0, t1, dt, method=euler_step_system):
    """Solve the ODE du/dt = f(u, t) from t0 to t1 with initial condition u0.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation du/dt = f(u, t).
    u0 : numpy.ndarray
        Initial condition.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    dt : float
        Time-step size.
    method : function, optional
        Numerical method to use to solve the ODE.

    Returns
    -------
    u : numpy.ndarray
        Array of approximate solutions at each time-step.
    t : numpy.ndarray
        Array of time points corresponding to the time-steps.
    """
    # Number of time-steps.
    N = int(np.ceil((t1 - t0)/dt))

    # Initialise the arrays to store the solution and time.
    u = np.zeros((N + 1, len(u0)))
    t = np.zeros(N + 1)

    # Set the initial condition.
    u[0] = u0
    t[0] = t0

    # Time loop.
    for n in range(N):
        t[n + 1] = t[n] + dt
        u[n + 1] = method(f, u[n], t[n], dt)

    return u, t

# %% [markdown]
# solve the 2nd order ODE $\ddot{x} = -x$ whi|ch is equivalent
# to the 1st order system $\dot{x} = y$, $\dot{y} = -x$.

# %%
# Solve the 2nd order ODE. (by solving the 1st order system)

# first order system


def f(u, t):
    x, y = u
    return np.array([y, -x])


# initial conditions
x0 = 1.0
y0 = 0.0
u0 = np.array([x0, y0])

# time interval
t0 = 0.0
t1 = 20.0

# time-step size
dt = 0.1

# solve the system
euler_u, euler_t = solve_to_system(f, u0, t0, t1, dt, method=euler_step_system)
rk4_u, rk4_t = solve_to_system(f, u0, t0, t1, dt, method=rk4_step_system)
print("Euler method: ", euler_u[-1])
print("RK4 method: ", rk4_u[-1])

# %% [markdown]
# Plot the results. What should the true solutions be?
# What goes wrong with the numerical solutions if you run them over a large range of $t$?
# (This is clearer if you plot $x$ against $\dot{x}$ rather than $x$ against $t$ and if you use timesteps that are not very small.)

# %%
# Plot the results

results = [euler_u, rk4_u]
labels = ['Euler', 'RK4']
for i in range(len(results)):
    plt.plot(results[i][:, 0], results[i][:, 1], label=labels[i])
plt.xlabel('x')
plt.ylabel('$\dot{x}$')
plt.legend()
plt.show()

# %% [markdown]
# The true solutions are $x = \cos(t)$ and $\dot{x} = -\sin(t)$.
# The Euler method is unstable for large $t$ because the error grows exponentially.
# The RK4 method is stable for large $t$ because the error grows quadratically.

# %% [markdown]
# Bonus points: implement some other methods rather than just Euler and RK4. There are loads of different 1-step integration methods.

# %%
# Bonus method: Heun's method


def heun_step(f, u, t, dt):
    """Compute the solution at the next time-step using Heun's method.

    Parameters
    ----------
    f : function
        Right-hand side of the differential equation du/dt = f(u, t).
    u : numpy.ndarray
        Solution at the previous time-step.
    t : float
        The current time.
    dt : float
        Time-step size.

    Returns
    -------
    u_n_plus_1 : numpy.ndarray
        Approximate solution at the next time-step.
    """
    u_n_plus_1 = u + dt*(f(u, t) + f(u + dt*f(u, t), t + dt))/2
    return u_n_plus_1


x0 = 1.0
t0 = 0.0
t1 = 1.0
deltat_max = 0.1
def func(x, t): return x


heun_u, heun_t = solve_to(func, x0, t0, t1, deltat_max, method=heun_step)
print("Heun method: ", heun_u[-1])
