from maddpg import MADDPG
from sim_env import UAVEnv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import coverage

from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载 OpenMP

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_velocity_magnitude(time_steps, velocities_magnitude):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_magnitude)):
        if i != 3:
            plt.plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_magnitude[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("Magnitude")
    plt.title("UAV Velocity Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocity_x(time_steps, velocities_x):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_x)):
        if i != 3:
            plt.plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_x[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_x$")
    plt.title("UAV $Vel_x$")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocity_y(time_steps, velocities_y):
    plt.figure(figsize=(15, 4))
    for i in range(len(velocities_y)):
        if i != 3:
            plt.plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            plt.plot(time_steps, velocities_y[i], label='Target')
    plt.xlabel("Time Steps")
    plt.ylabel("$vel_y$")
    plt.title("UAV $Vel_y$")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_velocities(velocities_magnitude, velocities_x, velocities_y):
    time_steps = range(len(velocities_magnitude[0]))
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    for i in range(len(velocities_magnitude)):
        if i != 3:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'UAV {i}')
        else:
            axs[0].plot(time_steps, velocities_magnitude[i], label=f'Target')
    axs[0].set_title('Speed Magnitude vs Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Speed Magnitude')
    axs[0].legend()

    for i in range(len(velocities_x)):
        if i != 3:
            axs[1].plot(time_steps, velocities_x[i], label=f'UAV {i}')
        else:
            axs[1].plot(time_steps, velocities_x[i], label=f'Target')
    axs[1].set_title('Velocity X Component vs Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Velocity X Component')
    axs[1].legend()

    for i in range(len(velocities_y)):
        if i != 3:
            axs[2].plot(time_steps, velocities_y[i], label=f'UAV {i}')
        else:
            axs[2].plot(time_steps, velocities_y[i], label=f'Target')
    axs[2].set_title('Velocity Y Component vs Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity Y Component')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    # ... [原有初始化代码] ...

    # 生成敏感度矩阵
    sensitivity_matrix = coverage.generate_sensitivity_function(grid_size=(100, 100), random=True, smooth_factor=10.0)

    # 创建figure并绘制热力图背景
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(sensitivity_matrix.T, origin='lower', cmap='viridis', alpha=0.4, extent=[0, 100, 0, 100])
    plt.colorbar(ax=ax, label='Environmental Sensitivity')


    # 初始化动画
    def update(frame, maddpg_agents=None):
        global obs, velocities_magnitude, velocities_x, velocities_y

        # ... [原有速度记录代码] ...

        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)

        obs_, _, dones = env.step(actions)

        # 清除当前绘图内容（但保留背景）
        ax.clear()
        # 重新绘制背景（必须！否则会被清除）
        ax.imshow(sensitivity_matrix.T, origin='lower', cmap='viridis', alpha=0.4, extent=[0, 100, 0, 100])
        # 绘制无人机位置
        env.render_anime(frame, ax=ax)  # 修改render_anime以接受ax参数

        obs = obs_
        if any(dones):
            ani.event_source.stop()
            print("Round-up finished in", frame, "steps.")
        return []


    # 需要修改 env.render_anime 以在指定ax上绘图
    def render_anime(self, frame, ax):
        # 在ax上绘制无人机位置
        for i in range(self.num_agents):
            x, y = self.multi_positions[i]
            ax.scatter(x, y, color='red', zorder=10)  # zorder确保在背景上层
        ax.set_title(f"Time Step: {frame}")


    ani = animation.FuncAnimation(fig, update, frames=10000, interval=20, blit=False)
    plt.show()