from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

path = './runs/shield_free_slicing/'

event_acc = EventAccumulator(path)
event_acc.Reload()

# Extracting scalar data
tags = event_acc.Tags()['scalars']
data = []

for tag in tags:
    events = event_acc.Scalars(tag)
    step = [event.step for event in events]
    value = [event.value for event in events]
    data.append({'tag': tag, 'step': step, 'value': value})

validation_offset = 3

# Creating a Pandas DataFrame
df = pd.DataFrame(data)
print(df)
loss_train = np.array(df['value'][0])
train_step = np.array(df['step'][0])
loss_validation = np.array(df['value'][1])
validation_step = np.array(df['step'][1])
plt.plot(train_step,loss_train, label='обучение', color = '#FF3333')
plt.plot(validation_step[validation_offset:],loss_validation[validation_offset:], label='валидация')
plt.legend()
plt.semilogy()
plt.xlabel('Итерация', color='white')
plt.ylabel('MSE', color='white')
# Display the DataFrame
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('white')
plt.gca().spines['bottom'].set_color('white')
plt.tick_params(axis='x', colors='white')
plt.tick_params(axis='y', colors='white')
plt.minorticks_off()
plt.savefig('./figs/try.png', transparent=True)