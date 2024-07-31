import multiprocessing
# from rllab.sampler.stateful_pool import singleton_pool
import scipy.spatial
import random
# from rllab.misc import logger
import sys
import os.path as osp

import matplotlib as mpl

mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc
#from curriculum.state.evaluator import parallel_map, disable_cuda_initializer

def disable_cuda_initializer(*args, **kwargs):
    import os
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

def parallel_map(func, iterable_object, num_processes=-1):
    """Parallelized map function based on python process
    Args:
    func: Pickleable callable object that takes one parameter.
    iterable_object: An iterable of elements to map the function on.
    num_processes: Number of process to use. When num_processes is 1,
                   no new process will be created.
    Returns:
    The list resulted in calling the func on all objects in the original list.
    """
    if num_processes == 1:
        return [func(x) for x in iterable_object]
    # if num_processes == -1:
    #     from rllab.sampler.stateful_pool import singleton_pool
    #     num_processes = singleton_pool.n_parallel
    process_pool = multiprocessing.Pool(
        num_processes,
        initializer=disable_cuda_initializer
    )
    results = process_pool.map(func, iterable_object)
    process_pool.close()
    process_pool.join()
    return results

class StateCollection(object):
    """ A collection of states, with minimum distance threshold for new states. """

    def __init__(self, max_len, distance_threshold=None, states_transform = None, idx_lim=None):
        self.distance_threshold = distance_threshold
        self.state_list = []
        self.states_transform = states_transform
        self.idx_lim = idx_lim
        self.max_len = max_len
        if self.states_transform:
            self.transformed_state_list = []

    @property
    def size(self):
        return len(self.state_list)

    def empty(self):
        self.state_list = []

    def sample(self, size, replace=False, replay_noise=0):
        states = sample_matrix_row(np.array(self.state_list), size, replace)
        if replay_noise > 0:
            states += replay_noise * np.random.randn(*states.shape)
        return states

    def append(self, states, n_process=None):
        if self.states_transform:
            return self.append_states_transform(states)
        if len(states) > 0:
            states = np.array(states)
            # logger.log("we are trying to append states: {}".format(states.shape))
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states = self._process_states(states)
            # logger.log("after processing, we are left with : {}".format(states.shape))
            if n_process is None:
                # n_process = singleton_pool.n_parallel
                n_process = 1
            elif n_process in [-1, 0]:
                n_process = 1
            if states.shape[0] >= n_process > 1:
                # states_per_process = states.shape[0] // singleton_pool.n_parallel
                states_per_process = states.shape[0] // 1
                list_of_states = [states[i * states_per_process: (i+1) * states_per_process, :] for i in range(n_process-1)]
                list_of_states.append(states[states_per_process * (n_process - 1):, :])
                states = parallel_map(self._select_states, list_of_states)
                states = np.concatenate(states)
            else:
                states = self._select_states(states)
            self.state_list.extend(states.tolist())
            # max length 
            self.state_list = self.state_list[len(self.state_list) - self.max_len:]
            return states

    def _select_states(self, states):
        # print('selecting states from shape: ', states.shape)
        selected_states = states
        selected_states_idx_lim = np.array([state[:self.idx_lim] for state in states])
        # print('selecting states from shape (after idx_lim of ', self.idx_lim, ': ', selected_states_idx_lim.shape)
        state_list_idx_lim = np.array([state[:self.idx_lim] for state in self.state_list])
        # print('the state_list_idx_lim shape: ', np.shape(self.state_list))
        if self.distance_threshold is not None and self.distance_threshold > 0:
            if len(self.state_list) > 0:
                dists = scipy.spatial.distance.cdist(state_list_idx_lim, selected_states_idx_lim)
                indices = np.amin(dists, axis=0) > self.distance_threshold
                selected_states = selected_states[indices, :]
        # print('the selected states are: {}'.format(selected_states.shape))
        return selected_states

    def _process_states(self, states):
        "keep only the states that are at more than dist_threshold from each other"
        # adding a states transform allows you to maintain full state information while possibly disregarding some dim
        states = np.array(states)
        results = [states[0]]
        results_idx_lim = [states[0][:self.idx_lim]]
        for i, state in enumerate(states[1:]):
            # print("analyzing state : ", i)
            if np.amin(scipy.spatial.distance.cdist(results_idx_lim, state.reshape(1, -1)[:self.idx_lim])) > self.distance_threshold:
                results.append(state)
                results_idx_lim.append(state[:self.idx_lim])
        return np.array(results)

    def _process_states_transform(self, states, transformed_states):
        "keep only the states that are at more than dist_threshold from each other"
        # adding a states transform allows you to maintain full state information while possibly disregarding some dim
        results = [states[0]]
        transformed_results = [transformed_states[0]]
        for i in range(1, len(states)):
            # checks if valid in transformed space
            if np.amin(scipy.spatial.distance.cdist(transformed_results, transformed_states[i].reshape(1, -1))) > self.distance_threshold:
                results.append(states[i])
                transformed_results.append(transformed_states[i])
        return np.array(results), np.array(transformed_results)

    def append_states_transform(self, states):
        assert self.idx_lim is None, "Can't use state transform and idx_lim with StateCollection!"
        if len(states) > 0:
            states = np.array(states)
            transformed_states = self.states_transform(states)
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states, transformed_states = self._process_states_transform(states, transformed_states)
                if len(self.state_list) > 0:
                    print("hi")
                    dists = scipy.spatial.distance.cdist(self.transformed_state_list, transformed_states)
                    indices = np.amin(dists, axis=0) > self.distance_threshold
                    states = states[indices, :]
                    transformed_states = transformed_states[indices, :]
            self.state_list.extend(states)
            self.transformed_state_list.extend(transformed_states)
            assert(len(self.state_list) == len(self.transformed_state_list))
        return states # modifed to return added states

    # def append(self, states):
    #     if self.states_transform:
    #         return self.append_states_transform(states)
    #     if len(states) > 0:
    #         states = np.array(states)
    #         if self.distance_threshold is not None and self.distance_threshold > 0:
    #             states = self._process_states(states)
    #             if len(self.state_list) > 0:
    #                 dists = scipy.spatial.distance.cdist(self.state_list, states)
    #                 indices = np.amin(dists, axis=0) > self.distance_threshold
    #                 states = states[indices, :]
    #         self.state_list.extend(states)
    #     return states # modifed to return added states

    @property
    def states(self):
        return np.array(self.state_list)

class SmartStateCollection(StateCollection):
    # should be used same as before, just need to update Q values
    #TODO: update alpha smartly
    def __init__(self, eps = 0.5, alpha = 0.3, abs = True, *args, **kwargs):
        self.eps = eps # percentage of random
        self.alpha = alpha
        self.abs = abs
        self.q_vals = {} #TODO: smarter way to keep list sorted, maybe priority queue?
        # use heapq https://stackoverflow.com/questions/7197315/5-maximum-values-in-a-python-dictionary if slow
        self.prev_vals = {}
        super(SmartStateCollection, self).__init__(*args, **kwargs)

    def update_starts(self, states, rewards, only_good = True, logger = None):
        old_states, old_rewards, new_states, new_rewards = [], [], [], []
        for i in range(len(states)):
            state = states[i]
            reward = rewards[i]
            if only_good:
                # TODO: set option
                # intuition is that we don't want states that we already master
                if reward < 0.02 or reward > 0.98:
                    continue
            # check if state shows up
            if len(self.state_list) > 0 and np.sum(np.all(state == self.state_list, axis = 1)) > 0:
                old_states.append(state)
                old_rewards.append(reward)
            else:
                new_states.append(state)
                new_rewards.append(reward)
        if logger is not None:
            logger.log("Total states: {}  New states: {}".format(len(states), len(new_states)))
        self.append(new_states, new_rewards)
        self.update_q(old_states, old_rewards)

    def append(self, states, rewards):
        zero_index = 0 # check on np.argmax
        added_states = super(SmartStateCollection, self).append(states)
        for state in added_states:
            index = np.argmax(np.all(states == state, axis =1)) # should only get one index
            if index == 0:
                zero_index += 1
            reward = rewards[index]
            self.q_vals[tuple(state)] = self.alpha * reward # TODO: not sure what the initialization should be, is there alpha term?
            self.prev_vals[tuple(state)] = reward
        assert (zero_index < 2)

    def sample(self, size, replace=False, replay_noise=0):
        size_random_samples = int(size * self.eps)
        size_good_samples = size - size_random_samples
        print("Random starts: {}".format(size_random_samples))
        states = sample_matrix_row(np.array(self.state_list), size_random_samples, replace)
        if size_good_samples == 0:
            return states # fully uniform states
        if self.abs:
            good_states = sorted(self.q_vals, key= lambda k: abs(self.q_vals[k]), reverse=True)[:size_good_samples]
        else:
            good_states = sorted(self.q_vals, key=self.q_vals.get, reverse=True)[:size_good_samples]
        good_states = np.array(good_states)
        return np.concatenate((states, good_states))
        # if replay_noise > 0:
        #     states += replay_noise * np.random.randn(*states.shape)
        # return states

    def update_q(self, states, rewards):
        # updated should be true if there are enough samples
        previous_values = np.array([self.prev_vals[tuple(state)] for state in states])
        curr_q_values =  np.array([self.q_vals[tuple(state)] for state in states])
        improvement = rewards - previous_values
        new_values = self.alpha * improvement + (1 - self.alpha) * curr_q_values

        for i in range(len(states)):
            # if updated[i]:
            self.q_vals[tuple(states[i])] = new_values[i]
            self.prev_vals[tuple(states[i])] = rewards[i]

def sample_matrix_row(M, size, replace=False):
    if size > M.shape[0]:
        return M
    if replace:
        indices = np.random.randint(0, M.shape[0], size)
    else:
        indices = np.random.choice(M.shape[0], size, replace=replace)
    return M[indices, :]