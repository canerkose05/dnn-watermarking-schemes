# Watermarking Deep Neural Networks

This repository contains **three different watermarking schemes for deep neural networks**, implemented during my HiWi position.

## What is DNN Watermarking?
Model watermarking embeds ownership information into a neural network such that the owner can later verify authorship or authenticate the model, even after deployment or model modification.

## Implemented Schemes
- [NeuNAC](https://www.sciencedirect.com/science/article/pii/S0020025521006642)
- [RIGA](https://arxiv.org/abs/1910.14268)
- [TATTOOED](https://arxiv.org/abs/2202.06091)

Each scheme is implemented independently and includes embedding and verification code.

## Structure
Each watermark folder contains:
- `WatermarkScheme.py` – Core implementation of the watermarking method
- `test.py` – embedding / extraction /verification steps

## Disclaimer
This is a re-implementation for demonstration purposes. No proprietary data or internal university code is included.
