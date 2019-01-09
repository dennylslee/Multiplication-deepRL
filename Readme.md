# Introduction 

Inspired by the paper in the reference section (Zarmeba 2016), this project's objective it to experiment with RL (reinforcement learning) and "teach" it to learn to do simple multiplication. 

The code structure is based on:
1. DDQN approach as documented in A. Oppermann tutorial (see acknowledgement section).  A few adaptations have to be incorporated since the learning requirements in this project, as compared to the "mountainCart",  are quite different. 
2. The code is set up for the following:
   - Two two digits multiplication setup (each number can range from 1 to 99)
   - The product of the two operands can be up to four digits
   - each digit can be between 0 and 9
   - each digit output by the RL agent is represented by a one-hot encoding format.

# Environment 

The environment setup between the RL agent (AI agent) is depicted below. 

![pics](https://github.com/dennylslee/Multiplication-deepRL/blob/master/Multiplication-game-env-setup.png)

The general strategy for the RL learning is to learn one digit at a time starting with uniary digit (also called digit1 in the picture). 

1. rewards is +1 for a match and -0.2 for a miss
2. action is a predicted digit in one-hot format
3. state is a combination of:
   - the two random generated numbers
   - the current digit matching position (starting with unit "ones" position)
   - the learned digit (correctly matched digit)

# Neural Network within the RL learner

The NN within the RL agent ("on policy" learner) is based on the a fully meshed feed foward 3 layer networks.  Note that too many nodes may cause overfitting but increasing combination of product outcomes do demands more nodes per layer. 

Usage of dropout regulation does not seem to have any positive contribution to the result.

NOTE: remember to NOT use activation in the last layer.  Just let it run wide and the Qmax selection will take advantage of it.  I accidently left it in initially and it generates all sort of wreid effects with the softmax squashing.

# Results

NOTE: at the time of writing, this project is NOT considered a success. 

In order to simplify the learning, we start with single digit multiplication.  Even with this simplification, it is proven to be difficult given the small NN and the limitation of a full mesh NN. We started with restricting the random generator such that it generates only value of 1 or 2.  After some moderate success, we gradually increase the possible range of random numbers to value of 1, 2, 3 or 4. Thus the product can be 1(1x1), 2 (2x1 or 1x2), 3, 4, 6, 9. ... 16  That is, it is a 6 categories classification problem. 

A snipbit of what the results look like (this is the case in which the agent learned well and the prediction are mostly correct on first try)

```
EPISODE  230
First num: 4 Second num: 4 Product: 16 [0, 0, 1, 6]
Done status:  False [0, 0, 0, 6]
Done status:  True [0, 0, 1, 6]
predicted digits:  [0, 0, 1, 6]  total reward earned:  164.4 end step:  1
EPISODE  231
First num: 4 Second num: 2 Product: 8 [0, 0, 0, 8]
Done status:  True [0, 0, 0, 8]
predicted digits:  [0, 0, 0, 8]  total reward earned:  165.4 end step:  0
EPISODE  232
First num: 3 Second num: 1 Product: 3 [0, 0, 0, 3]
Done status:  True [0, 0, 0, 3]
predicted digits:  [0, 0, 0, 3]  total reward earned:  166.4 end step:  0
EPISODE  233
First num: 4 Second num: 4 Product: 16 [0, 0, 1, 6]
Done status:  False [0, 0, 0, 6]
Done status:  True [0, 0, 1, 6]
predicted digits:  [0, 0, 1, 6]  total reward earned:  168.4 end step:  1
EPISODE  234
First num: 1 Second num: 4 Product: 4 [0, 0, 0, 4]
Done status:  True [0, 0, 0, 4]
predicted digits:  [0, 0, 0, 4]  total reward earned:  169.4 end step:  0
EPISODE  235
First num: 2 Second num: 1 Product: 2 [0, 0, 0, 2]
Done status:  True [0, 0, 0, 2]
predicted digits:  [0, 0, 0, 2]  total reward earned:  170.4 end step:  0
EPISODE  236
First num: 4 Second num: 3 Product: 12 [0, 0, 1, 2]
Done status:  False [0, 0, 0, 8]
Done status:  False [0, 0, 0, 2]
Done status:  True [0, 0, 1, 2]
predicted digits:  [0, 0, 1, 2]  total reward earned:  172.2 end step:  2
EPISODE  237
First num: 4 Second num: 3 Product: 12 [0, 0, 1, 2]
Done status:  False [0, 0, 0, 8]
Done status:  False [0, 0, 0, 2]
Done status:  True [0, 0, 1, 2]
predicted digits:  [0, 0, 1, 2]  total reward earned:  174.0 end step:  2
EPISODE  238
First num: 2 Second num: 4 Product: 8 [0, 0, 0, 8]
Done status:  True [0, 0, 0, 8]
predicted digits:  [0, 0, 0, 8]  total reward earned:  175.0 end step:  0
EPISODE  239
First num: 1 Second num: 4 Product: 4 [0, 0, 0, 4]
Done status:  True [0, 0, 0, 4]
predicted digits:  [0, 0, 0, 4]  total reward earned:  176.0 end step:  0

```

The Qmax per epsiode curve is plotted along with the per-epsiode and accumulated rewards are plotted in the following dashboard.  Note that a positive prediction yield a +1 reward whereas each failed prediction yield a negative (e.g. -0.2) in this result below (up to digit value of 4 and 4 as inputs)

![pics2](https://github.com/dennylslee/Multiplication-deepRL/blob/master/Results_figure.png)

A much larger NN was used as the policy learner here to deal with a digit value of 5 and 5 as inputs. 

![pics3](https://github.com/dennylslee/Multiplication-deepRL/blob/master/Results_figure2.png)

# The deficiency of the current design and future work

1. The NN is simply treating this as a multi-class classification problem.  It is effectively performing a pattern recognition of numbers based on observing the state and earning the rewards when matches occur.  Possible future enhancement is to use a LSTM based RNN such that sequential information can be learned. 
 - the reference (Zaremba 2016) also cited that multipication algorithm requires memory but a feed forward NN has no inherent memory capability. 
2. The state space design does not facilate effective learning. Better state space design improvement is required. 

For a good summary of why RL is difficult; particular around choice of algo, policy learning,  reward design, convergence, etc.), please see the well written [blog post](https://www.alexirpan.com/2018/02/14/rl-hard.html) on "Deep Reinforcement Learning Doesn't Work Yet" by alexirpan.

# Acknowledgement 

Much of the code is adopted from A. Oppermann's blog in Medium. It is an excellent tutorial with detailed walk through. You can find it [here](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47).


# Reference

[1] Zaremba, et. al. "Learning Simple Algorithms from Examples", 2016