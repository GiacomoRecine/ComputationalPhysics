# three_body_adaptive_qdraw_precise_square_window.py
import numpy as np
from math import sqrt
from qdraw import window, circle, draw, show

def acceleration_three(r, m, G=1.0, eps=1e-10):
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(3):
            if i != j:
                rij = r[j] - r[i]
                dist2 = np.dot(rij, rij) + eps
                dist = sqrt(dist2)
                a[i] += G * m[j] * rij / (dist**3)
    return a

# ------------------ RK4 STEP ------------------
def rk4_step_three(r, v, dt, m, G=1.0):
    def deriv(r_local, v_local):
        dr = v_local
        dv = acceleration_three(r_local, m, G)
        return dr, dv

    k1_r, k1_v = deriv(r, v)
    k2_r, k2_v = deriv(r + 0.5*dt*k1_r, v + 0.5*dt*k1_v)
    k3_r, k3_v = deriv(r + 0.5*dt*k2_r, v + 0.5*dt*k2_v)
    k4_r, k4_v = deriv(r + dt*k3_r, v + dt*k3_v)

    r_next = r + (dt/6.0)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    v_next = v + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return r_next, v_next

# ------------------Adaptive integration------------------
def integrate_three_adaptive(r0, v0, m, G=1.0, t_end=10.0,
                             dt0=1e-3, delta=1e-3, min_dt=1e-8, max_steps=int(1e6)):
    r = r0.copy()
    v = v0.copy()
    t = 0.0
    dt = dt0

    r_hist = [r.copy()]
    dt_hist = [dt]

    step = 0
    while t < t_end and step < max_steps:
        r1, v1 = rk4_step_three(r, v, dt, m, G)
        r_half, v_half = rk4_step_three(r, v, dt/2, m, G)
        r2, v2 = rk4_step_three(r_half, v_half, dt/2, m, G)

        err = np.max(np.linalg.norm(r2 - r1, axis=1))
        rho = 30 * delta * dt / (err + 1e-40)

        if rho >= 1.0:
            r, v = r2, v2
            t += dt
            r_hist.append(r.copy())
            dt_hist.append(dt)
            dt *= min(1.5, rho**0.25)
        else:
            dt *= max(0.5, rho**0.25)

        if dt < min_dt:
            print("Warning: dt troppo piccolo, fermo l'integrazione")
            break

        step += 1

    return np.array(r_hist), np.array(dt_hist)

# ------------------ Animation ------------------
def animate_three_bodies_qdraw(r_hist, dt_hist, masses, colors=["red","green","blue"],
                               C=0.1, dt_max=1e-2):
   
    x_min, x_max = np.min(r_hist[:,:,0])-2, np.max(r_hist[:,:,0])+2
    y_min, y_max = np.min(r_hist[:,:,1])-2, np.max(r_hist[:,:,1])+2
    
    side = max(x_max - x_min, y_max - y_min)
    x_center = 0.5*(x_max + x_min)
    y_center = 0.5*(y_max + y_min)
    xlim = (x_center - side/2, x_center + side/2)
    ylim = (y_center - side/2, y_center + side/2)

    w = window(width=1200, height=1200, xlim=xlim, ylim=ylim,
               title="Three-body problem", bgcolor="white")

    stars = []
    for i in range(3):
        pos = tuple(r_hist[0,i])
        size = 0.5 + 0.003*masses[i]   
        c = circle(size=size, pos=pos, color=colors[i])
        c.trail(True, length=500, width=1, color=colors[i])
        stars.append(c)

    for i, frame in enumerate(r_hist):
        for j, s in enumerate(stars):
            s.setpos(tuple(frame[j]))
        dt_frame = min(C/dt_hist[i], dt_max)
        draw(dt_frame)

    show()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    # Masses (all equal)
    m = np.array([1.0, 1.0, 1.0])

    # Moore–Chenciner–Montgomery initial positions
    r0 = np.array([[0.0, 0.0],
                   [0.97000436, -0.24308753],
                   [-0.97000436, 0.24308753]])

    # Moore–Chenciner–Montgomery initial velocities
    v0 = np.array([[0.93240737, 0.86473146],
                   [-0.46620369, -0.43236573],
                   [-0.46620369, -0.43236573]])

    # Adaptive integration
    r_hist, dt_hist = integrate_three_adaptive(r0, v0, m, t_end=80.0,
                                               dt0=1e-3, delta=1e-4)

    # Animation
    animate_three_bodies_qdraw(r_hist, dt_hist, masses=m)