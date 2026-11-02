"""
Сравнение методов расчёта градиента при градиентном спуске
для функции двух переменных.

Методы:
  1. Аналитический
  2. Наивный численный (центральные разности, фиксированный шаг)
  3. Стохастический численный (случайный шаг дифференцирования)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter


OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def f(x, y):
    """Целевая функция: f(x, y) = 2*sin(3xy) + 4*cos(x + 4y)."""
    return 2 * np.sin(3 * x * y) + 4 * np.cos(x + 4 * y)


def grad_analytical(x, y):
    """Аналитический градиент."""
    df_dx = 6 * y * np.cos(3 * x * y) - 4 * np.sin(x + 4 * y)
    df_dy = 6 * x * np.cos(3 * x * y) - 16 * np.sin(x + 4 * y)
    return np.array([df_dx, df_dy])


def grad_numerical_naive(x, y, delta=0.001):
    """Центральные конечные разности с фиксированным шагом delta."""
    df_dx = (f(x + delta, y) - f(x - delta, y)) / (2 * delta)
    df_dy = (f(x, y + delta) - f(x, y - delta)) / (2 * delta)
    return np.array([df_dx, df_dy])


def grad_numerical_stochastic(x, y, delta=0.001):
    """Центральные разности со случайным шагом delta_x, delta_y."""
    delta_min = delta / 2.0
    delta_max = 3.0 * delta

    delta_x = np.random.uniform(delta_min, delta_max)
    delta_y = np.random.uniform(delta_min, delta_max)

    df_dx = (f(x + delta_x, y) - f(x - delta_x, y)) / (2 * delta_x)
    df_dy = (f(x, y + delta_y) - f(x, y - delta_y)) / (2 * delta_y)
    return np.array([df_dx, df_dy])


def gradient_descent(x0, y0, grad_func, lambda_val=0.01, max_iter=500, tol=1e-6):
    """Градиентный спуск с сохранением траектории."""
    trajectory = [[x0, y0]]
    x, y = x0, y0

    for _ in range(max_iter):
        grad = grad_func(x, y)
        if np.linalg.norm(grad) < tol:
            break

        x = x - lambda_val * grad[0]
        y = y - lambda_val * grad[1]
        trajectory.append([x, y])

    return np.array(trajectory)


def visualize_descent(trajectories, method_name, lambda_val, show=False):
    """3D-поверхность, контур и тепловая карта с траекториями спуска."""
    fig = plt.figure(figsize=(15, 5))

    all_x = np.concatenate([traj[:, 0] for traj in trajectories])
    all_y = np.concatenate([traj[:, 1] for traj in trajectories])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    x_margin = (x_max - x_min) * 0.2 or 0.5
    y_margin = (y_max - y_min) * 0.2 or 0.5
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin

    x_range = np.linspace(x_min, x_max, 200)
    y_range = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.6)

    for i, traj in enumerate(trajectories):
        z_traj = f(traj[:, 0], traj[:, 1])
        ax1.plot(
            traj[:, 0], traj[:, 1], z_traj,
            ".-", linewidth=2, markersize=3, alpha=0.7, label=f"Траектория {i + 1}",
        )
        ax1.scatter(traj[0, 0], traj[0, 1], z_traj[0], c="green", s=100, marker="o",
                    edgecolors="black", linewidths=2)
        ax1.scatter(traj[-1, 0], traj[-1, 1], z_traj[-1], c="red", s=150, marker="*",
                    edgecolors="black", linewidths=2)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("f(x, y)")
    ax1.set_title(f"3D: {method_name}\nλ={lambda_val}")

    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=30, cmap=cm.viridis)
    plt.colorbar(contour, ax=ax2)

    for traj in trajectories:
        ax2.plot(traj[:, 0], traj[:, 1], ".-", linewidth=2, markersize=3, alpha=0.7)
        ax2.scatter(traj[0, 0], traj[0, 1], c="green", s=100, marker="o",
                    edgecolors="black", linewidths=2, zorder=5)
        ax2.scatter(traj[-1, 0], traj[-1, 1], c="red", s=150, marker="*",
                    edgecolors="black", linewidths=2, zorder=5)

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(f"Контур: {method_name}\nλ={lambda_val}")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(133)
    im = ax3.imshow(
        Z, extent=[x_min, x_max, y_min, y_max],
        origin="lower", cmap=cm.viridis, aspect="auto",
    )
    plt.colorbar(im, ax=ax3)

    for traj in trajectories:
        ax3.plot(traj[:, 0], traj[:, 1], ".-", linewidth=2, markersize=3, alpha=0.7)
        ax3.scatter(traj[0, 0], traj[0, 1], c="green", s=100, marker="o",
                    edgecolors="black", linewidths=2, zorder=5)
        ax3.scatter(traj[-1, 0], traj[-1, 1], c="red", s=150, marker="*",
                    edgecolors="black", linewidths=2, zorder=5)

    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title(f"Тепловая карта: {method_name}\nλ={lambda_val}")
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)

    plt.tight_layout()
    safe_name = method_name.replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"gradient_descent_{safe_name}_lambda_{lambda_val}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Сохранено: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_method(name, grad_func, start_points, lambda_values, show=False):
    """Запуск градиентного спуска для набора точек и шагов λ."""
    print(f"\n=== {name} ===")
    for lambda_val in lambda_values:
        trajectories = []
        for start in start_points:
            traj = gradient_descent(
                start[0], start[1], grad_func,
                lambda_val=lambda_val, max_iter=500,
            )
            trajectories.append(traj)
            print(
                f"Старт: [{start[0]:6.2f}, {start[1]:6.2f}] -> "
                f"Финиш: [{traj[-1, 0]:6.3f}, {traj[-1, 1]:6.3f}] | "
                f"Итераций: {len(traj):3d} | "
                f"f_min: {f(traj[-1, 0], traj[-1, 1]):7.4f}"
            )
        visualize_descent(trajectories, name, lambda_val, show=show)


def make_animation_gif(
    start_points,
    lambda_val=0.005,
    max_iter=120,
    n_frames=40,
    fps=12,
    filename="gradient_descent_animation.gif",
):
    """Короткий GIF: траектории аналитического градиентного спуска."""
    trajectories = [
        gradient_descent(s[0], s[1], grad_analytical, lambda_val=lambda_val, max_iter=max_iter)
        for s in start_points
    ]
    max_len = max(len(t) for t in trajectories)
    step = max(1, max_len // n_frames)
    frame_ids = list(range(0, max_len, step))
    if frame_ids[-1] != max_len - 1:
        frame_ids.append(max_len - 1)

    all_x = np.concatenate([t[:, 0] for t in trajectories])
    all_y = np.concatenate([t[:, 1] for t in trajectories])
    pad = 0.4
    x_min, x_max = all_x.min() - pad, all_x.max() + pad
    y_min, y_max = all_y.min() - pad, all_y.max() + pad

    xs = np.linspace(x_min, x_max, 250)
    ys = np.linspace(y_min, y_max, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Gradient Descent  |  analytical  |  λ = {lambda_val}")
    cf = ax.contourf(X, Y, Z, levels=40, cmap=cm.viridis, alpha=0.95)
    plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="f(x, y)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    lines, heads = [], []
    for i, c in enumerate(colors):
        (ln,) = ax.plot([], [], "-", color=c, linewidth=1.8, alpha=0.9)
        (hd,) = ax.plot([], [], "o", color=c, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.6)
        lines.append(ln)
        heads.append(hd)
        ax.scatter(
            trajectories[i][0, 0], trajectories[i][0, 1],
            c="lime", s=55, edgecolors="black", zorder=5,
        )

    iter_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=11, color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55),
    )

    def update(frame_idx):
        k = frame_ids[frame_idx]
        for i, traj in enumerate(trajectories):
            end = min(k + 1, len(traj))
            lines[i].set_data(traj[:end, 0], traj[:end, 1])
            heads[i].set_data([traj[end - 1, 0]], [traj[end - 1, 1]])
        iter_text.set_text(f"iteration: {k}")
        return lines + heads + [iter_text]

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=80, blit=False)
    out_path = os.path.join(OUTPUT_DIR, filename)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Сохранено: {out_path}")
    return out_path



if __name__ == "__main__":
    start_points = [
        [2.5, 2.5],
        [-2.0, 1.5],
        [1.0, -2.0],
        [-1.5, -1.5],
        [0.5, 0.5],
        [2.0, -1.0],
        [-2.5, 2.0],
        [1.5, 1.5],
        [3.0, 3.0],
        [-3.0, -3.0],
    ]

    lambda_values = [0.001, 0.005, 0.01]
    delta = 0.001
    show_plots = False  # True - открывать окна с графиками

    print("=== Генерация GIF-анимации ===")
    make_animation_gif(start_points[:8], lambda_val=0.005)

    run_method("Аналитический", grad_analytical, start_points, lambda_values, show=show_plots)

    print(f"\nНаивный численный: фиксированный Δ = {delta}")
    run_method(
        "Наивный численный",
        lambda x, y: grad_numerical_naive(x, y, delta=delta),
        start_points,
        lambda_values,
        show=show_plots,
    )

    print(f"\nСтохастический: Δ ∈ [{delta / 2}, {3 * delta}]")
    for lambda_val in lambda_values:
        trajectories = []
        print(f"\n=== Стохастический | λ={lambda_val} ===")
        for start in start_points:
            np.random.seed(42)
            traj = gradient_descent(
                start[0], start[1],
                lambda x, y: grad_numerical_stochastic(x, y, delta=delta),
                lambda_val=lambda_val,
                max_iter=500,
            )
            trajectories.append(traj)
            print(
                f"Старт: [{start[0]:6.2f}, {start[1]:6.2f}] -> "
                f"Финиш: [{traj[-1, 0]:6.3f}, {traj[-1, 1]:6.3f}] | "
                f"Итераций: {len(traj):3d} | "
                f"f_min: {f(traj[-1, 0], traj[-1, 1]):7.4f}"
            )
        visualize_descent(trajectories, "Стохастический", lambda_val, show=show_plots)

    print("\n" + "=" * 70)
    print("Все графики сохранены в папку assets/")
    print(f"  Базовый Δ = {delta}")
    print(f"  Стохастический Δ_min = {delta / 2}, Δ_max = {3 * delta}")
    print("=" * 70)
