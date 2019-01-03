# Introduction 

Inspired by the paper in reference section (Zarmeba 2016), this project's objective it to experiment with RL (reinforcement learning) and "teach" it to learn to do simple multiplication. 

The code structure is based on:
1. DDQN approach as documented in A. Oppermann tutorial (see acknowledgement section).  A few adaptation has been to incorporated since the learning in this project versus the "mountainCart" is quite different. 
2. The code are structure for a set up:
   - Two two digits multiplication setup (each number can range from 1 to 99)
   - The product of the two operands can be up to four digits
   - each digit can be between 0 and 9
   - each digit is represented by a one-hot encoding format.

# Environment 

The environment setup between the RL agent (AI agent) is depicted below. 

[pics](https://github.com/dennylslee/Multiplication-deepRL/blob/master/Multiplication-game-env-setup.png)

The general strategy for the learning is to learn one digit at a time starting with uniary digit (also called digit1 in the picture). 

1. rewards is +1 for a match and -0.2 for a miss
2. action is a predicted digit in one-hot format
3. state is a combination of:
   - the two random generated numbers
   - the current digit matching position (starting with uniary position)
   - the learned digit (correctly matched digit)

# Neural Network within the RL learner

The NN within the RL agent (on policy learner) is based on the a fully meshed 3 layer networks.  Note that too many nodes causes overfitting but increasing combination of product outcome do demands more nodes per layer. 

Usage of dropout regulation does not seem to have any positive contribution to the result.

# Results

NOTE: at the time of writing, this project is NOT considered a success. 

In order to simplify the learning, we start with single digit multiplication.  Even that is proven to be difficult given the small NN and the limitation of a full mesh NN. We started with restricting the random generator to generate either value of 1 or 2.  After some moderate success, we gradually increase the possible range of random number to value of 1, 2, or 3. Thus the product can be 1(1x1), 2 (2x1 or 1x2), 3, 4, 6, 9.  That is, it is a 6 categories classification problem. 


# The deficient of the current design and future work

1. The NN is simply treating this as a multi-class classification problem.  It is effectively performing a pattern recognition of numbers based on observing the state and earning the rewards when matches occur.  Possible future enhancement is to use a LSTM based RNN such that sequential information can be learned. 
2. The state space design does not facilate effective learning. Better state space design improvement is required. 

# Acknowledgement 

Much of the code is adopted from A. Oppermann's blog in Medium. It is an excellent tutorial with detailed walk through. You can find it [here](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47).


# Reference

[1] Zaremba, et. al. "Learning Simple Algorithms from Examples", 2016