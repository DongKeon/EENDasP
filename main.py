import os
import pdb
import glob
import argparse
import numpy as np
from itertools import combinations, permutations


# utils
def read_rttm_file_subsmpling(rttm_name, subsampling=1):
    resolution = int(100 / subsampling)
    lines = []
    if os.path.exists(rttm_name):
        with open(rttm_name, 'r') as fp:
            lines = fp.readlines()
    
    speakers = {}
    uttr = []
    max_len = 0
    n_spk = 0
    for line in lines:
        line = line.replace('\n', '')
        data = line.split(' ')
        while '' in data:
            data.remove('')
        start = int(float(data[3]) * resolution)
        end = start + int(float(data[4]) * resolution)
        spk_name = data[7]
        if not spk_name in speakers.keys():
            speakers[spk_name] = n_spk
            n_spk += 1
        uttr.append((start, end, speakers[spk_name]))
        max_len = max(max_len, end)

    arr = np.zeros((n_spk, max_len + resolution), dtype=np.int32)
    for (start, end, uttr_id) in uttr:
        arr[uttr_id, start:end] = 1

    return arr

def zero_padding(x, length):
    t, f = x.shape
    x_pad = np.zeros([length, f], dtype=x.dtype)
    x_pad[:t, :] = x
    return x_pad

def thresholding(arr, threshold=0.5):
    return np.where(arr > threshold, 1, 0)
    
def select_active_frames(arr):
    active_frames = np.where(arr.sum(axis=0) > 0)[0]
    return active_frames

def speech_similarity(U, V):
    return len(np.intersect1d(U, V))

def non_speech_similarity(U, V, T):
    return len(np.intersect1d(np.setdiff1d(T, U), np.setdiff1d(T, V)))

def calculate_similarity(U, V, T):
    return speech_similarity(U, V) + non_speech_similarity(U, V, T)

def make_rttm_from_EENDasP(wf, session, processed_arr, frame_shift, subsampling, sampling_rate):
    for spkid, frames in enumerate(processed_arr):
        frames = np.pad(frames, (1, 1), 'constant')
        changes, = np.where(np.diff(frames, axis=0) != 0)
        fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
        for s, e in zip(changes[::2], changes[1::2]):
            print(fmt.format(
                    session,
                    s * frame_shift * subsampling / sampling_rate,
                    (e - s) * frame_shift * subsampling / sampling_rate,
                    str(spkid)), file=wf)
                    #session + "_" + str(spkid)), file=wf)


class EENDasP_example():
    def __init__(self):
        # xvector
        self.xvec_arr = np.zeros([3, 12])
        self.xvec_arr[0][1:3] = 1
        self.xvec_arr[1][3:6] = 1
        self.xvec_arr[2][0] = 1
        self.xvec_arr[2][6:9] = 1
        self.xvec_arr[2][11] = 1
        self.S_xvec, self.T_xvec = self.xvec_arr.shape
        self.spk_pairs = list(combinations(range(self.S_xvec), 2))
        self.total_spk_set = set(np.arange(self.S_xvec))
        self.total_frames = np.arange(self.T_xvec)

        # EEND
        self.__call__()

    def frame_selection(self, spk_pair):
        other_spks = list(self.total_spk_set - set(spk_pair))
        other_spks_frames = select_active_frames(self.xvec_arr[other_spks, :])
        selected_frames = np.setdiff1d(self.total_frames, other_spks_frames)
        return selected_frames

    def decide_processing_order(self):
        selected_frames_list = []
        for spk_pair in self.spk_pairs:
            selected_frames = self.frame_selection(spk_pair)
            selected_frames_list.append(selected_frames)

        pair_len_list = [len(selected_frames_list[i]) for i in range(len(self.spk_pairs))]
        order = np.argsort(pair_len_list)[::-1] # dscending order

        return order

    def eend_processing(self, i):
        if i == 0:
            ys = np.zeros((2, 10))
            ys[0][1:5] = 1
            ys[1][0] = 1
            ys[1][3:7] = 1
            ys[1][9] = 1
        elif i == 1:
            ys = np.zeros((2, 8))
            ys[0][0:2] = 1
            ys[0][3:5] = 1
            ys[0][-1] = 1
            ys[1][1:3] = 1
            ys[1][-1] = 1
        elif i == 2:
            ys = np.zeros((2, 5))
            ys[0][0:2] = 1
            ys[1][1:3] = 1
        return ys

    def solve_permutation(self, T_i, T_j, eend):
        T = self.total_frames
        spk_permutes = list(permutations(range(2), 2))

        similarity = None
        for spk_permute in spk_permutes:
            Q_u = np.where(eend[spk_permute,:][0] == 1)[0]
            Q_v = np.where(eend[spk_permute,:][1] == 1)[0]
            similarity_permute = calculate_similarity(Q_u, T_i, T) + \
                                 calculate_similarity(Q_v, T_j, T)
            if similarity is None or similarity < similarity_permute:
                similarity = similarity_permute
                T_hat_i = Q_u
                T_hat_j = Q_v

        return T_hat_i, T_hat_j

    def check_condition(self, P_ij, T_i, T_hat_i, alpha=0.5):
        """
        P_ij: Selected frames
        T: Previous predicted frames 
        T_hat: New results
        alpha: Lower limit of the ratio of the intersection between 
               the new results (T_hat) and the previous results (T n P_ij)
        """
        prev_results = np.intersect1d(T_i, P_ij)
        ratio_intersect = len(np.intersect1d(T_hat_i, prev_results)) / len(prev_results)
        return True if ratio_intersect > alpha else False

    def update(self, T_i, T_j, T_hat_i, T_hat_j, P_ij, spk_pair):
        """
        K: estimated number of speakers
        """
        K = self.S_xvec

        if self.check_condition(P_ij, T_i, T_hat_i):
            if K == 2:
                T_i = self.simple_update(T_hat_i, T, P_ij)
            if K >= 3:
                T_i = self.fully_update(T_i, T_hat_i, T_hat_j)

        if self.check_condition(P_ij, T_j, T_hat_j):
            if K == 2:
                T_j = self.simple_update(T_hat_j, T, P_ij)
            if K >= 3:
                T_j = self.fully_update(T_j, T_hat_i, T_hat_j)

        self.xvec_arr[spk_pair[0], T_i] = 1
        self.xvec_arr[spk_pair[1], T_j] = 1

    def simple_update(self, T_hat_i, T, P_ij):
        return np.union1d(T_hat_i, np.setdiff1d(T, P_ij))

    def fully_update(self, T_i, T_hat_i, T_hat_j):
        return np.union1d(T_i, np.intersect1d(T_hat_i, T_hat_j))

    def __call__(self):
        """
        P_ij: Selected frame
        """
        order = self.decide_processing_order()
        T = self.total_frames
        
        for i in range(len(self.spk_pairs)): # descending order
            idx = np.where(order == i)[0][0]
            spk_pair = self.spk_pairs[idx]
            P_ij = self.frame_selection(spk_pair)

            T_i = np.where(self.xvec_arr[spk_pair[0]] == 1)[0]
            T_j = np.where(self.xvec_arr[spk_pair[1]] == 1)[0]

            selected_eend = self.eend_processing(i)
            eend_arr = np.zeros_like(self.xvec_arr[spk_pair,:])
            eend_arr[:, P_ij] = selected_eend

            T_hat_i, T_hat_j = self.solve_permutation(T_i, T_j, eend_arr)
            self.update(T_i, T_j, T_hat_i, T_hat_j, P_ij, spk_pair)
            print(self.xvec_arr)
            pdb.set_trace()
            

if __name__ == '__main__':
    EENDasP_example()
