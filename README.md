# Haggles

Project Haggles is my open source solution for kaggle competetions.

## Structure

The repository structured by kaggle competitions name, every competition contains `data, model` and `src`.

```
├── README.md
├── compstat
│   ├── data
│   ├── model
│   └── src
├── compstat2
│   ├── data
│   ├── model
│   └── src
└── compstat3
    ├── data
    ├── model
    └── src
```

## Environment

- Keras2 + TensorFlow (backend) + Scikit-Learn + Numpy + Pandas + h5py + matplotlib ...

```bash
virtualenv -p python3 .env
source .env/bin/activate
pip install -r req.txt
```

### Running

1. Pub data in `<competition_name>/data/`;
2. Go `src` folder, run `main.py`

```python
cd competition/<competition_name>/src
python main.src # with virtualenv activate
```

### Benchmarks

|         Dataset         | Test Accuracy |             Competition Name             |
| :---------------------: | :-----------: | :--------------------------------------: |
|          MNIST          |      99%      |      Arch as same as Fashion-MNIST       |
|      Fashion-MNIST      |      94%      | [compstat](https://www.kaggle.com/c/compstat) |
| Caltech-101-silhouettes |      77%      | [compstat2](https://www.kaggle.com/c/compstat2) |
|        CIFAR-10         |      85%      | [compstat3](https://www.kaggle.com/c/compstat3) |

## License

The [MIT](LICENSE) License (MIT) Copyright © [2017-present] Ou Changkun, https://changkun.de

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
