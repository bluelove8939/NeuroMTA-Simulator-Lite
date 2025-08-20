# NeuroMTA Simulator

## Introduction

NeuroMTA is a highly programmable cycle-level multi-tile deep learning accelerator simulator. This simulator provides a fundamental framework to implement various multi-tile accelerator architectures and programming API to create test workload with inter-core spatial dataflow. The simulator is implemented as a Python library and easy to be extended by the hardware and software developers. 

## Installation

```bash
conda create -n neuromta python=3.11
conda activate neuromta
pip install -r requirements.txt
pip install -e .
```