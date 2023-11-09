# DL4G - Deep Learning for Games

## Project Summary
Code for the project in the DL4G module at HSLU. The goal of the project is to build a bot that can play the game of [Jass](https://en.wikipedia.org/wiki/Jass). Various methods of game play were explored. Rule-based, MCTS, DMCTS, DNN-based.
Our bot utilizes DMCTS and a DNN network trained on trump selection data.

## Card Play
To tackle Jass's imperfect information, the bot employs DMCTS with multiple determinizations processed in parallel across all CPU cores within a 30-second decision window.

![Gameplay Screenshot](https://github.com/theoliver7/hslu-dl4g/assets/10463395/c9fc170f-747f-4717-b921-4a1e0846e225)

## Trump Selection
The DNN used for trump selection is a six-layer dense network, incorporating relu activations, regularization via L2, and dropout layers to mitigate overfitting, culminating in a softmax layer for final decision output.


