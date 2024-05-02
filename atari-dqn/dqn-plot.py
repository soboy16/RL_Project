import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_learning_curve(log_dir, smooth_window=10):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get all scalar data
    scalar_data = {
        key: event_acc.Scalars(key) for key in ('charts/episodic_return', 'charts/epsilon')
    }

    # Extract steps and values
    steps = {key: [event.step for event in events] for key, events in scalar_data.items()}
    values = {key: [event.value for event in events] for key, events in scalar_data.items()}

    # Smooth data
    smoothed_values = {key: moving_average(data, smooth_window) for key, data in values.items()}
    smoothed_steps = {key: steps[key][:len(smoothed_values[key])] for key in steps}

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps['charts/episodic_return'], values['charts/episodic_return'], label='Actual')
    plt.plot(smoothed_steps['charts/episodic_return'], smoothed_values['charts/episodic_return'], 'r-', label='Moving Average')
    #plt.plot(steps['charts/epsilon'], values['charts/epsilon'], label='Epsilon')
    #plt.plot(smoothed_steps['charts/epsilon'], smoothed_values['charts/epsilon'], 'r--', label='Epsilon (Smoothed)')
    plt.xlabel('Timesteps')
    plt.ylabel('Episodic Return')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    log_dir = "runs/BreakoutNoFrameskip-v4__dqn_atari__1__1714053325/events.out.tfevents.1714053325.DESKTOP-QG4VMUB.18052.0"  # Change this to the directory where your TensorBoard logs are stored
    plot_learning_curve(log_dir)
