[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" /> <img alt="Keras" src="https://img.shields.io/badge/Keras%20-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FLearning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Learning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras

Learning rate schedules aim to change the learning rate during neural netowrk training by lowering the `lr` according to a predefined functions/timetable. There are number of Learning Rate Schedulers availbel some of the popular ones are,

* Step Decay
* Exponential Decay
* Cosine Decay
* K-Decay
* Polynomial Decay

**Some more advanced Learning-Rate-Schedulers are,**
* Exponential Decay with Burnin
* SGDR 
This SGDR further has two varients,

1. STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
2. STOCHASTIC GRADIENT DESCENT WITH WARMUP

#### For [Loss-Functions-Package-Tensorflow-Keras-PyTorch](https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch/)
#### For [Evaluation-Metrics-Package-Tensorflow-PyTorch-Keras](https://github.com/Mr-TalhaIlyas/Evaluation-Metrics-Package-Tensorflow-PyTorch-Keras/)

### Step Decay
Drop learning rate after certain epochs (i.e., `drop_epoch`) by a factor of `lr_decay`.

#### Implementation, Hyperparamenters and Constants
```python
drop_epoch = 3
lr_decay = 0.85
```

```python
def step_decay(epoch, initial_lr, lr_decay, drop_epoch):
    initial_lrate = initial_lr
    drop = lr_decay
    epochs_drop = drop_epoch
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
```

### Exponential Decay
Drop learning rate exponentially.

#### Implementation, Hyperparamenters and Constants
```python
k = 0.1
```

```python
def exp_decay(epoch, initial_lr, Epoch):
    k = 0.1
    lrate = initial_lr * np.exp(-k*epoch)
    return lrate
```

### Cosine Decay
A learning rate schedule that uses a cosine decay schedule
details [here](https://arxiv.org/abs/1608.03983)
#### Implementation, Hyperparamenters and Constants
```python
alpha=0.0
```

```python
def cosine_decay(epoch, initial_lr, Epoch):
    alpha=0.0
    epoch = min(epoch, Epoch)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / Epoch))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_lr * decayed
# Equivelant to,
tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps, alpha=0.0)
```

### K-Decay
A new LR schedule.with a new hyper-parameter k controls the change degree of LR, whereas the original method of k at 1.
details [here](https://arxiv.org/abs/2004.05909)
#### Implementation, Hyperparamenters and Constants
```python
k = 3
N = 4
```

```python
def K_decay(t = x,L0=inint_lr,Le=final_lr,T=Epoch,N=N,k=k):
    lr = (L0 - Le) * (1 - t**k / T**k)**N + Le
    return lr
```

### Polynomial Decay
A Polynomial Decay policy.
details [here](https://ieeexplore.ieee.org/document/8929465)
#### Implementation, Hyperparamenters and Constants
```python
power = 0.9
```

```python
def polynomial_decay(epoch, initial_lr, Epoch, power):
    initial_lrate = initial_lr
    lrate = initial_lrate * math.pow((1-(epoch/Epoch)),power)
    return lrate
```
## Usage
For all above LR schedules you can create a custom function callback as follows,
Here I combined 3 schedules (`step`, `poly` and `k`) from the above list in one callback
```python
class CustomLearningRateScheduler(Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule, initial_lr, lr_decay, total_epochs, drop_epoch, power):
        #super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.total_epochs = total_epochs
        self.drop_epoch = drop_epoch
        self.power = power
        
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
            
        if self.schedule == 'step_decay':
            self.schedule = step_decay
        if self.schedule == 'polynomial_decay':
            self.schedule = polynomial_decay
        if self.schedule == 'K_decay':
            self.schedule = K_decay
            
        lr = self.initial_lr
        if lr is None:
            # Get the current learning rate from model's optimizer.
            lr = float(K.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr, self.lr_decay, self.drop_epoch, self.total_epochs, self.power)
        # Set the value back to the optimizer before this epoch starts
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {}".format(epoch+1, scheduled_lr))
```
Now for `polynomial_decay` call in `main` as
```python
LR_schedule = CustomLearningRateScheduler(polynomial_decay, initial_lr, lr_decay, Epoch, drop_epoch, power)
```

## PyTorch Implementation

```python

import math

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['lr'] > 0: optimizer.param_groups[i]['lr'] = lr
            # optimizer.param_groups[0]['lr'] = lr
            # for i in range(1, len(optimizer.param_groups)):
            #     optimizer.param_groups[i]['lr'] = lr * 10

```
### Usage

``` python
scheduler = LR_Scheduler(config['lr_schedule'], config['learning_rate'], config['Epoch'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

for epoch in range(config['Epoch']):
    for step, data_batch in enumerate(train_data):
        # update learning rate in optimizer
        scheduler(optimizer, step, epoch)
        # train code here
```
## Visualization
```python
# just for curve visulaization
x = np.arange(0,Epoch) # current epoch
k_d = []
for i in range(len(x)):
    z = K_decay(t = i,L0=inint_lr,Le=final_lr,T=Epoch,N=N,k=k) # select any funciton here
    k_d.append(z)
plt.plot(x, k_d, 'g', label = 'K_decay')
```
![alt-text](https://github.com/Mr-TalhaIlyas/Learning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras/blob/main/screens/img1.png)

## SGDR
See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent with Warm Restarts. https://arxiv.org/abs/1608.03983
### STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
See the code and comments for details
```python
class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic/warm restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())
        
    def on_epoch_begin(self, epoch, logs=None):
        print(60*'=')
        print("Epoch %05d: Learning rate is %6.2e"  % (epoch+1, K.get_value(self.model.optimizer.lr)))
        
    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
```
### **Usage**
```python
LR_schedule = SGDRScheduler(min_lr=1e-7, max_lr=initial_lr, steps_per_epoch=num_images/Batch_size,
                                lr_decay=lr_decay,cycle_length=cycle,mult_factor=mul_factor)
```
#### Visual Curve
![alt-text](https://github.com/Mr-TalhaIlyas/Learning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras/blob/main/screens/sgdlr.png)

### STOCHASTIC GRADIENT DESCENT WITH WARMUP
See the code and comments for details
```python
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        print('\nBatch %05d: setting learning rate to %s.' % (self.global_step + 1, lr))
```

### **Usage**
```python
LR_schedule = WarmUpCosineDecayScheduler(learning_rate_base=initial_lr,
                                         total_steps=int(Epoch * num_images/Batch_size),
                                         warmup_learning_rate=0.0,
                                         warmup_steps=int(warmup_epoch * num_images/Batch_size))
```
#### Visual Curve
![alt-text](https://github.com/Mr-TalhaIlyas/Learning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras/blob/main/screens/lrw.png)
#### Warmup Cosine Decay Schedular by subclassing `tf.keras.optimziers.schedules.LearningRateSchedule`

```python
class WarmupCosineDecayLRScheduler(
  tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, 
                  max_lr: float,
                  warmup_steps: int,
                  decay_steps: int,
                  alpha: float = 0.) -> None:
        super(WarmupCosineDecayLRScheduler, self).__init__()

        self.name = 'WarmupCosineDecayLRScheduler'
        self.alpha = alpha

        self.max_lr = max_lr
        self.last_step = 0

        self.warmup_steps = int(warmup_steps)
        self.linear_increase = self.max_lr / float(self.warmup_steps)

        self.decay_steps = int(decay_steps)

    def _decay(self):
        rate = tf.subtract(self.last_step, self.warmup_steps) 
        rate = tf.divide(rate, self.decay_steps)
        rate = tf.cast(rate, tf.float32)

        cosine_decayed = tf.multiply(tf.constant(math.pi), rate)
        cosine_decayed = tf.add(1., tf.cos(cosine_decayed))
        cosine_decayed = tf.multiply(.5, cosine_decayed)

        decayed = tf.subtract(1., self.alpha)
        decayed = tf.multiply(decayed, cosine_decayed)
        decayed = tf.add(decayed, self.alpha)
        return tf.multiply(self.max_lr, decayed)

    def __call__(self, step):
      self.last_step = step
      lr_s = tf.cond(
                    tf.less(self.last_step, self.warmup_steps),
                    lambda: tf.multiply(self.linear_increase, self.last_step),
                    lambda: self._decay())
      return lr_s

    def get_config(self) -> dict:
        config = {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha
        }
        return config
```
#### Uage
 
```python
LR_schedule = WarmupCosineDecayLRScheduler(max_lr=initial_lr,
                                          decay_steps=int(Epoch * (train_paths)/batch_size),
                                          warmup_steps=int(warmup_epoch * (train_paths)/batch_size))
optimizer = tf.keras.optimizers.Adam(LR_schedule, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)                                         
```
### Exponential Decay with Warmstart

Introduced in [Transformers](https://arxiv.org/abs/1706.03762) used for [ViT](https://arxiv.org/abs/2010.11929)

```python
class ExponentialDecaywithWarmstart(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model=128, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    #print(step)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```
#### Usage

```python 
LR_schedule = ExponentialDecaywithWarmstart(d_model=2048)
optimizer = tf.keras.optimizers.Adam(LR_schedule, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)     
```
#### Visual Curve
![alt-text](https://github.com/Mr-TalhaIlyas/Learning-Rate-Schedulers-Packege-Tensorflow-PyTorch-Keras/blob/main/screens/tf_lr.png)

### Exponential Decay with Burnin
In this schedule, learning rate is fixed at burnin_learning_ratefor a fixed period, before transitioning to a regular exponential decay schedule.

⚠ Still a work in progress.

**Numpy**
```python
def exp_burnin_decay(burnin_epoch, burnin_lr, epoch, initial_lr, Epoch):
    if epoch <= burnin_epoch:
        lrate = burnin_lr
        initial_lr = lrate
    else:
        k = 0.1
        lrate = initial_lr * np.exp(-k*(epoch))
    return lrate
```
**Tensorflow**
```python
import tensorflow as tf

def exponential_decay_with_burnin(global_step,
                                  learning_rate_base,
                                  learning_rate_decay_steps,
                                  learning_rate_decay_factor,
                                  burnin_learning_rate=0.0,
                                  burnin_steps=0,
                                  min_learning_rate=0.0,
                                      staircase=True):
    """Exponential decay schedule with burn-in period.
    
    In this schedule, learning rate is fixed at burnin_learning_rate
    for a fixed period, before transitioning to a regular exponential
    decay schedule.
    
    Args:
      global_step: int tensor representing global step.
      learning_rate_base: base learning rate.
      learning_rate_decay_steps: steps to take between decaying the learning rate.
        Note that this includes the number of burn-in steps.
      learning_rate_decay_factor: multiplicative factor by which to decay
        learning rate.
      burnin_learning_rate: initial learning rate during burn-in period.  If
        0.0 (which is the default), then the burn-in learning rate is simply
        set to learning_rate_base.
      burnin_steps: number of steps to use burnin learning rate.
      min_learning_rate: the minimum learning rate.
      staircase: whether use staircase decay.
    
    Returns:
      If executing eagerly:
        returns a no-arg callable that outputs the (scalar)
        float tensor learning rate given the current value of global_step.
      If in a graph:
        immediately returns a (scalar) float tensor representing learning rate.
    """
    if burnin_learning_rate == 0:
      burnin_learning_rate = learning_rate_base
    
    """Callable to compute the learning rate."""
    post_burnin_learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step - burnin_steps,
        learning_rate_decay_steps,
        learning_rate_decay_factor,
        staircase=staircase)
    if callable(post_burnin_learning_rate):
      post_burnin_learning_rate = post_burnin_learning_rate()
      
    return tf.maximum(tf.where(
        tf.less(tf.cast(global_step, tf.int32), tf.constant(burnin_steps)),
        tf.constant(burnin_learning_rate),
        post_burnin_learning_rate), min_learning_rate, name='learning_rate')
```

## Test Code for Checking Learning Rate Callbacks
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import numpy as np

sample_count = num_images
data = np.random.random((sample_count, 100))
labels = np.random.randint(10, size=(sample_count, 1))

# Convert labels to categorical one-hot encoding.
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics='accuracy', )

model.fit(data, one_hot_labels, epochs=Epoch, batch_size=Batch_size,
        verbose=1, callbacks=[LR_schedule])
```
