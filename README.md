# Accelerating ReLU for MPC-Based Private Inference with a Communication-Efficient Sign Estimation (MLSys '24)
![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
--------------------------------------------------------------------------------

Code repo for HummingBird, described in the following paper:

Kiwan Maeng and G. Edward Suh. Accelerating ReLU for MPC-Based Private Inference with a Communication-Efficient Sign Estimation. MLSys, 2024.

## How to run

1. Install the patched CrypTen and dependencies.
```bash
pip install -r requirements.txt
cd CrypTen/
pip install -r requirements.txt
python3 setup.py install
```

2. See `scripts/` for example executions.

## Citation
If you use HummingBird, please cite us:
```
@inproceedings{
  author={Maeng, Kiwan and Suh, G. Edward},
  title={Accelerating ReLU for MPC-Based Private Inference with a Communication-Efficient Sign Estimation},
  booktitle={Proceedings of Machine Learning and Systems (MLSys)},
  year={2024},
}
```

## License
HummingBird is MIT licensed, as found in the LICENSE file.
