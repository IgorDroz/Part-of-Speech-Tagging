from feature_builder import feature_builder
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
from time import time
import numpy as np
import pickle

class MEMM_trainer(feature_builder):

	def __init__(self, file_statistics, threshold):
		super(MEMM_trainer,self).__init__(file_statistics,threshold)

	def fit(self, lambda_=0.005,save=True):
		start = time()
		w_0 = np.zeros(self._num_total_features, dtype=np.float32)
		optimal_params = fmin_l_bfgs_b(func=self.calc_objective_per_iter, x0=w_0, args=[lambda_], maxiter=1000, iprint=1)
		print('training took {} seconds'.format(time()-start))

		if save:
			weights_path = 'weights_'+self._file_statistics.file_path.rstrip('.wtag')+'.pkl'
			with open(weights_path, 'wb') as f:
				pickle.dump(optimal_params, f)

	def calc_objective_per_iter(self, w_i, lambda_):
		"""
			Calculate max entropy likelihood for an iterative optimization method
			:param w_i: weights vector in iteration i
			:param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

				The function returns the Max Entropy likelihood (objective) and the objective gradient
		"""
		linear_term = self.linear_term(self._sparse_features_matrix, w_i)
		normalization_term = self.normalization_term(self._histories_dict, w_i, self._all_possible_sparse_matrix)
		regularization = self.regularization_term(lambda_, w_i)
		empirical_counts = self.empirical_counts(self._sparse_features_matrix)
		expected_counts = self.expected_counts(self._histories_dict,self._num_total_features, w_i, self._all_possible_sparse_matrix,self._all_possible_sparse_matrix_trans)
		regularization_grad = self.reg_grad(lambda_, w_i)
		likelihood = linear_term - normalization_term - regularization
		grad = empirical_counts - expected_counts - regularization_grad

		return (-1) * likelihood, (-1) * grad

	@staticmethod
	def linear_term(matrix_features, w):
		return sum(matrix_features.dot(w))

	@staticmethod
	def normalization_term(historys, w, sparse_matrix_all_pos_tags_list):
		sum_log = 0
		for i, history in enumerate(historys):
			sum_log += np.log(np.sum(np.exp(sparse_matrix_all_pos_tags_list[i].dot(w))))
		return sum_log

	@staticmethod
	def regularization_term(lambda_, w):
		return 0.5 * lambda_ * LA.norm(w)

	@staticmethod
	def empirical_counts(sparse_matrix_historys_features):
		return sparse_matrix_historys_features.sum(axis=0)

	@staticmethod
	def expected_counts(historys, result_length, w, sparse_matrix_all_pos_tags_list,sparse_matrix_all_pos_tags_list_trans):
		result = np.zeros((1, result_length))
		for i, history in enumerate(historys):
			exp_multiplication = np.exp(sparse_matrix_all_pos_tags_list[i].dot(w))
			prob_array = exp_multiplication / sum(exp_multiplication)
			result += sparse_matrix_all_pos_tags_list_trans[i].dot(prob_array)
		return result

	@staticmethod
	def reg_grad(lambda_, w):
		return w * lambda_
