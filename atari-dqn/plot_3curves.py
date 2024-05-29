#import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

def smooth_data(data, smooth_window):
    smoothed_data = []
    for i in range(len(data)):
        start = max(0, i - smooth_window // 2)
        end = min(len(data), i + smooth_window // 2)
        smoothed_data.append(np.mean(data[start:end]))
    return smoothed_data

def plot_learning_curves(log_dirs, labels, smooth_window=500):
    plt.figure(figsize=(10, 6))
    
    colors = ['#4b82db', '#6cf757', '#f76457']   # Add more colors if you have more datasets
    
    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Get all scalar data
        scalar_data = {
            key: event_acc.Scalars(key) for key in ('charts/episodic_return',)
        }

        # Extract steps and values
        steps = {key: [event.step for event in events] for key, events in scalar_data.items()}
        values = {key: [event.value for event in events] for key, events in scalar_data.items()}

        # Smooth data
        smoothed_values = {key: smooth_data(data, smooth_window) for key, data in values.items()}
        smoothed_steps = {key: steps[key][:len(smoothed_values[key])] for key in steps}

        # Plot
        color = colors[i % len(colors)]
        plt.plot(steps['charts/episodic_return'], values['charts/episodic_return'], color=color, alpha=0.1, label=f'{label} Actual')
        plt.plot(smoothed_steps['charts/episodic_return'], smoothed_values['charts/episodic_return'], color=color, label=f'{label} Smoothed')

    plt.xlabel('Episodes')
    plt.ylabel('Episodic Return')
    plt.title('Space Invaders')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    log_dirs = [
        "runs/SpaceInvadersNoFrameskip-v4__dqn_atari__1__1716712607/events.out.tfevents.1716712607.DESKTOP-QG4VMUB.23340.0",
        "runs/SpaceInvadersNoFrameskip-v4__expsarsa__1__1716791260/events.out.tfevents.1716791260.LAPTOP-KLSLFGQE.17484.0",
        "runs/SpaceInvadersNoFrameskip-v4__sarsa_bayesian__1__1716564263/events.out.tfevents.1716564263.LAPTOP-KLSLFGQE.27968.0"
    ]
    labels = [
        "DQN Atari",
        "Expected SARSA",
        "SARSA Bayesian"
    ]
    plot_learning_curves(log_dirs, labels)
