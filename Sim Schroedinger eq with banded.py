import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from banded import banded   # importa il solver

#===============================================================
#  Function for A banded
#===============================================================
def get_A_band(h, hbar, m, a, N):
    
    a1 = 1 + 1j*hbar*h/(2*m*a*a)
    a2 = -1j*hbar*h/(4*m*a*a)

    A_band = np.zeros((3, N-1), dtype=complex)

    A_band[1, :] = a1        # principak diagonal
    A_band[0, 1:] = a2       # superior diagonal 
    A_band[2, :-1] = a2      # inferior diagonal

    return A_band, a1, a2


#===============================================================
#  Construction of B
#===============================================================
def get_B_diagonals(h, hbar, m, a):
    b1 = 1 - 1j*hbar*h/(2*m*a*a)
    b2 = 1j*hbar*h/(4*m*a*a)
    return b1, b2


#===============================================================
#  initial wave function psi
#===============================================================
def psi_0(x0, k, sigma, x):
    return np.exp(1j*k*x) * np.exp(-((x - x0)**2) / (2*sigma**2))


#===============================================================
#  Crack-Nicolson method 
#===============================================================
def crank_nicolson_step(psi, A_band, b1, b2):
    psi_inner = psi[1:-1]

    # v = B psi
    v = b1*psi_inner + b2*(psi[2:] + psi[:-2])

    # Risoluzione A ψ_new = v
    psi_new_inner = banded(A_band, v, up=1, down=1)

    psi_new = np.zeros_like(psi, dtype=complex)
    psi_new[1:-1] = psi_new_inner
    return psi_new


#===============================================================
#  temporal evolution
#===============================================================
def crank_nicolson(psi_initial, A_band, b1, b2, num_steps):
    psi = psi_initial.copy()
    evolution = [psi.copy()]

    for _ in range(num_steps):
        psi = crank_nicolson_step(psi, A_band, b1, b2)
        evolution.append(psi.copy())

    return np.array(evolution)


#===============================================================
#  problem parameter
#===============================================================
L = 1e-8
N = 1000
a = L/N
x = np.linspace(0, L, N+1)

h = 1e-18
hbar = 1.0545718e-34
m = 9.10938356e-31

x0 = L/2
sigma = 1e-10
k = 5e10

#===============================================================
#  matrix construction
#===============================================================
A_band, a1, a2 = get_A_band(h, hbar, m, a, N)
b1, b2 = get_B_diagonals(h, hbar, m, a)

psi_initial = psi_0(x0, k, sigma, x)

num_steps = 800

#===============================================================
#  evolution
#===============================================================
psi_time = crank_nicolson(psi_initial, A_band, b1, b2, num_steps)


fig, ax = plt.subplots(figsize=(8,4))
line, = ax.plot(x, np.real(psi_time[0]), color='blue')

ax.set_ylim(-np.max(np.abs(psi_time[0]))*1.5, np.max(np.abs(psi_time[0]))*1.5)
ax.set_xlabel("x (m)")
ax.set_ylabel("Re(ψ(x,t))")
ax.set_title("Temporal evolution of the real part of wavefunction")
#ax.grid(True)

#function for animation
def update(frame):
    line.set_ydata(np.real(psi_time[frame]))  # real part
    #ax.set_title(f"t = {frame*h:.2e} s")
    ax.set_title("Temporal evolution of the real part of wavefunction")
    return line

#animation
ani = FuncAnimation(fig, update, frames=len(psi_time), interval=20, blit=False)

plt.show()

plt.plot(x, np.real(psi_time[0]), color='blue')
plt.xlabel("x (m)")
plt.ylabel("Re(ψ(x,0))")
plt.title("Initial wavefunction (real part)")
plt.show()
