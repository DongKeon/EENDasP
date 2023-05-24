# EENDasP: End-to-End Speaker Diarization as Post-Processing

This repository contains an implementation of the **EENDasP**: "End-to-End Speaker Diarization as Post-Processing" approach, as presented in *Proc. ICASSP*, 2021. For more information about the methodology and experiment results, please refer to the [official paper](https://arxiv.org/abs/2012.10055).

## Installation

To install and run this code, you will need Python 3.6 or later. You will also need to install `numpy` if you haven't done so already. You can install the necessary libraries by running the following command:

```sh
pip install numpy
```

## Usage

You can run the main example by executing the `main.py` file from the command line:

```sh
python main.py
```

The `main.py` file contains an example class `EENDasP_example` demonstrating the key steps of the EENDasP approach. 

- **Initialization (`__init__`)**: Initializes some example `xvector` data and necessary variables.
- **Frame selection (`frame_selection`)**: Selects the frames that will be processed for a given speaker pair.
- **Processing order decision (`decide_processing_order`)**: Determines the order in which speaker pairs will be processed.
- **EEND processing (`eend_processing`)**: This is where the actual processing happens for each speaker pair, generating the diarization results. In this example, some dummy `ys` data is used.
- **Permutation solving (`solve_permutation`)**: Solves the permutation problem to match the EEND outputs to the original speakers.
- **Condition checking (`check_condition`)**: Checks whether the updated results (T_hat) should replace the previous results (T).
- **Update (`update`)**: Updates the `xvec_arr` for the current speaker pair based on the processing results.
- **Simple and Full Update (`simple_update`, `fully_update`)**: These methods implement the two update rules described in the paper, depending on the estimated number of speakers.

## Contributing

If you have any suggestions, bug reports, or annoyances please feel free to report them to the issue tracker of this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We would like to acknowledge the authors of the original paper for their contributions to the field and their clear presentation of the method, which greatly facilitated the creation of this example implementation.
