from MEMM_trainer import MEMM_trainer
import numpy as np
import pickle
from time import time


class MEMM_predictor(MEMM_trainer):

	def __init__(self, file_statistics, threshold):
		super(MEMM_predictor,self).__init__(file_statistics,threshold)
		self.pos_tags_viterbi = ["*"] + list(self._file_statistics.pos_tags)

	def predict(self,file_path,weights_path,beam_size,eval_mode=True ,plot =False):
		start = time()
		with open(weights_path, 'rb') as f:
			optimal_params = pickle.load(f)
		pre_trained_weights = optimal_params[0]
		sentences = []
		if eval_mode:
			with open(file_path) as f:
				for line in f:
					splited_words = line.split()
					words = []
					tags = []
					for word_idx in range(len(splited_words)):
						cur_word, cur_tag = splited_words[word_idx].split('_')
						words.append(cur_word)
						tags.append(cur_tag)
					words.append('*STOP*')
					sentences.append([words, tags])
		else:
			with open(file_path) as f:
				for line in f:
					words = line.split()+['*STOP*']
					sentences.append(words)

		self.predictions = []
		for sentence in sentences:
			if eval_mode:
				tags = self.MEMM_viterbi(pre_trained_weights, sentence[0], beam_size)
			else:
				tags = self.MEMM_viterbi(pre_trained_weights, sentence, beam_size)
			self.predictions.append(tags)

		print('inference took {} seconds'.format(time()-start))
		if eval_mode:
			return self.predictions,self.eval(self.predictions, sentences, plot)
		return self.predictions

	@staticmethod
	def eval(predictions, sentences ,plot=False):
		true_predict = 0
		words_sum = 0
		true_labels = []
		pred_labels = []

		for i, tags in enumerate(predictions):
			words_sum += len(tags)
			for j, tag in enumerate(tags):
				true_labels.append(sentences[i][1][j])
				pred_labels.append(tag)
				if sentences[i][1][j] == tag:
					true_predict += 1
		accuracy = (true_predict / words_sum) * 100
		print('accuracy is {}'.format(accuracy))

		if plot:
			from sklearn.metrics import confusion_matrix
			import pandas as pd
			import seaborn as sns
			import matplotlib.pyplot as plt
			np.set_printoptions(precision=2)
			labels = np.unique(pred_labels) if len(np.unique(pred_labels)) > len(
				np.unique(true_labels)) else np.unique(true_labels)
			cm = confusion_matrix(true_labels,pred_labels,labels=labels)
			most_error_indices = np.argsort(np.sum(cm*(np.ones((len(labels),)*2)-np.diag(np.ones((len(labels),)))),axis=1))[-10:]
			cm = cm[most_error_indices][:,most_error_indices]
			indices = labels[most_error_indices]
			df_cm = pd.DataFrame(cm, index=indices,columns=indices)
			plt.tight_layout()
			sns.heatmap(df_cm, annot=True,cmap=sns.cm.rocket_r)
			plt.ylabel('Ground truth')
			plt.xlabel('Predictions')
			plt.savefig('./confusion_matrix.jpg')
			plt.show()



	def MEMM_viterbi(self, w, sentence, beam_size):

		predictions, pi, back_pointers =[],[],[]
		pi_0 = np.zeros((len(self.pos_tags_viterbi),)*2)
		pi_0[0][0] = 1
		pi.append(pi_0)
		back_pointers_0 = np.full([len(self.pos_tags_viterbi),]*2, np.nan)
		back_pointers.append(back_pointers_0)
		u, t = 0,0

		curr_options_set=[(u,t)]
		for i in range(len(sentence) - 1):
			pi_i = np.zeros((len(self.pos_tags_viterbi),)*2)
			back_pointers_i = np.full([len(self.pos_tags_viterbi),]*2, np.nan)
			curr_word,next_word= sentence[i], sentence[i + 1]
			prev_word = sentence[i - 1] if i > 0 else '*'
			prev_x2_word = sentence[i - 2] if i > 1 else '*'

			for pair in curr_options_set:
				t,u = pair
				prev_x2_tag = self.pos_tags_viterbi[t]
				prev_tag = self.pos_tags_viterbi[u]
				for v,curr_tag in enumerate(self.pos_tags_viterbi):
					history = (prev_x2_word, prev_x2_tag, prev_word, prev_tag, curr_word, curr_tag, next_word, None)
					temp = pi[i][t][u] * self.q_calc(history, w)
					pi_i[u][v] = temp
					back_pointers_i[u][v] = t

			back_pointers.append(back_pointers_i)
			pi.append(pi_i)
			curr_options_set = np.dstack(np.unravel_index(np.argsort(pi_i.ravel()),pi[0].shape))[0][-beam_size:]

		last_max_index = np.where(pi[-1] == np.max(pi[-1]))
		u, v = last_max_index[0][0], last_max_index[1][0]
		result = []
		result.append(v)
		result.append(u)
		for idx in range(len(sentence) - 1, 0, -1):
			y_k = back_pointers[idx][int(result[-1])][int(result[-2])]
			result.append(int(y_k))
		for tag in result:
			predictions.append(self.pos_tags_viterbi[tag])

		return predictions[::-1][2:]

	def q_calc(self,history,w):
		try:
			prev_x2_word, prev_x2_tag, prev_word, prev_tag, curr_word, curr_tag, next_word, next_tag = history
		except:
			prev_x2_word, prev_x2_tag, prev_word, prev_tag, curr_word, curr_tag, next_tag = history

		features_vector = self.represent_input_with_features(history)
		history_all_pos_tags = {}
		for i, pos_tag in enumerate(self.pos_tags_viterbi):
			temp = [prev_x2_word, prev_x2_tag, prev_word, prev_tag, curr_word, pos_tag, next_word]
			history_all_pos_tags[tuple(temp)] = i
		features_matrix_all_pos_tag = self.create_features_sparse_matrix(history_all_pos_tags)
		multiplication = features_matrix_all_pos_tag.dot(w)
		exp_multiplication = np.exp(multiplication)
		denominator = sum(exp_multiplication)
		numerator = np.exp(sum([w[i] for i in features_vector]))
		q = numerator / denominator
		return q