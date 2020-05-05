from scipy.sparse import csr_matrix
from itertools import islice
import re

class feature_builder():

	def __init__(self, file_statistics, threshold):
		self._file_statistics = file_statistics  # statistics class, for each featue gives empirical counts
		self._threshold = threshold                    # feature count threshold - empirical count must be higher than this
		self.initiallize_variables()
		self.build_features()

		self.create_history_dict()
		self._sparse_features_matrix = self.create_features_sparse_matrix(self._histories_dict)
		self.create_history_all_pos_tags()

		for history_dict in self._all_tag_histories_list:
			self._all_possible_sparse_matrix.append(self.create_features_sparse_matrix(history_dict))

		for matrix in self._all_possible_sparse_matrix:
			self._all_possible_sparse_matrix_trans.append(matrix.transpose())


	def initiallize_variables(self):
		self._num_total_features = 0
		self._all_possible_sparse_matrix = []
		self._all_possible_sparse_matrix_trans = []

		# Init all features dictionaries
		self._words_tags_dict = {}
		self._suffixes_tags_dict = {}
		self._prefixes_tags_dict = {}
		self._trigram_tag_dict = {}
		self._bigram_tag_dict = {}
		self._unigram_tag_dict = {}
		self._first_capital_letter_tags_dict = {}
		self._capital_letter_word_tags_dict = {}
		self._next_word_tags_dict = {}
		self._prev_word_tags_dict = {}

	def build_features(self):
		self.define_f100()
		self.define_f101()
		self.define_f102()
		self.define_f103()
		self.define_f104()
		self.define_f105()
		self.define_f106()
		self.define_f107()
		self.define_first_capital_letter_tag_features_indices()
		self.define_company_feature_index()
		self.define_all_capital_letters_feature_index()
		self.define_is_number_feature_index()
		self.define_title_feature_index()
		self.define_common_adj_suffix_feature_index()
		self.define_plural_feature_index()
		self.define_plural_and_capital_feature_index()

	def define_f100(self):
		"""
			Extract out of text all word/tag pairs
			:param file_path: full path of the file to read
				return all word/tag pairs with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				#del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if (((word, tag) not in self._words_tags_dict.keys()) and (self._file_statistics.words_tags_count_dict[(word, tag)] >= self._threshold)):
						self._words_tags_dict[(word, tag)] = self._num_total_features
						self._num_total_features += 1

	def define_f106(self):
		"""
			Extract out of text all word/tag pairs
			:param file_path: full path of the file to read
				return all word/tag pairs with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				#del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx == (len(splited_words) - 1):
						next_word = '*STOP*'
					else:
						next_word = splited_words[idx + 1]
					if (((next_word, tag) not in self._next_word_tags_dict.keys()) and (self._file_statistics.next_word_tags_count_dict[(next_word, tag)] >= self._threshold)):
						self._next_word_tags_dict[(next_word, tag)] = self._num_total_features
						self._num_total_features += 1


	def define_f107(self):
		"""
			Extract out of text all word/tag pairs
			:param file_path: full path of the file to read
				return all word/tag pairs with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				#del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx == 0:
						prev_word = '*'
					else:
						prev_word = splited_words[idx - 1]
					if (((prev_word, tag) not in self._prev_word_tags_dict.keys()) and (self._file_statistics.prev_word_tags_count_dict[(prev_word, tag)] >= self._threshold)):
						self._prev_word_tags_dict[(prev_word, tag)] = self._num_total_features
						self._num_total_features += 1

	def define_f101(self):
		"""
		Extract out of text all suffixes/tag pairs from length <=4
		:param file_path: full path of file to read
			return all suffix/tag pairs with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					for suffix_length in range(1,5):
						if suffix_length==len(word):
							break
						suffix = word[-suffix_length:len(word)]
						if (((suffix, tag) not in self._suffixes_tags_dict.keys()) and (self._file_statistics.suffixes_tags_count_dict[(suffix, tag)] >= self._threshold)):
							self._suffixes_tags_dict[(suffix, tag)] = self._num_total_features
							self._num_total_features += 1

	def define_f102(self):
		"""
		Extract out of text all prefixes/tag pairs from length <=4
		:param file_path: full path of file to read
			return all prefix/tag pairs with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					for prefix_length in range(1,5):
						if prefix_length==len(word):
							break
						prefix = word[0:prefix_length]
						if (((prefix, tag) not in self._prefixes_tags_dict.keys()) and (self._file_statistics.prefixes_tags_count_dict[(prefix, tag)] >= self._threshold)):
							self._prefixes_tags_dict[(prefix, tag)] = self._num_total_features
							self._num_total_features += 1

	def define_f103(self):
		"""
		Extract out of text all three tag pairs
		:param file_path: full path of file to read
			return all three tag pairs with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx > 1:
						prev_word, prev_tag = splited_words[idx - 1].split('_')
						prev_x2_word, prev_x2_tag = splited_words[idx - 2].split('_')
						if (((prev_x2_tag, prev_tag, tag) not in self._trigram_tag_dict.keys()) and (self._file_statistics.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] >= self._threshold)):
							self._trigram_tag_dict[(prev_x2_tag, prev_tag, tag)] = self._num_total_features
							self._num_total_features += 1
					elif idx == 1:
						prev_word, prev_tag = splited_words[idx - 1].split('_')
						prev_x2_word, prev_x2_tag = '*', '*'
						if (((prev_x2_tag, prev_tag, tag) not in self._trigram_tag_dict.keys()) and (self._file_statistics.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] >= self._threshold)):
							self._trigram_tag_dict[(prev_x2_tag, prev_tag, tag)] = self._num_total_features
							self._num_total_features += 1
					elif idx == 0:
						prev_word, prev_tag = '*','*'
						prev_x2_word, prev_x2_tag = '*', '*'
						if (((prev_x2_tag, prev_tag, tag) not in self._trigram_tag_dict.keys()) and (self._file_statistics.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] >= self._threshold)):
							self._trigram_tag_dict[(prev_x2_tag, prev_tag, tag)] = self._num_total_features
							self._num_total_features += 1


	def define_f104(self):
		"""
		Extract out of text all two tag pairs
		:param file_path: full path of file to read
			return all two tag pairs with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx > 0:
						prev_word, prev_tag = splited_words[idx - 1].split('_')
						if (((prev_tag, tag) not in self._bigram_tag_dict.keys()) and (self._file_statistics.bigram_tags_count_dict[(prev_tag, tag)] >= self._threshold)):
							self._bigram_tag_dict[(prev_tag, tag)] = self._num_total_features
							self._num_total_features += 1
					else:
						prev_word, prev_tag = '*', '*'
						if (((prev_tag, tag) not in self._bigram_tag_dict.keys()) and (self._file_statistics.bigram_tags_count_dict[(prev_tag, tag)] >= self._threshold)):
							self._bigram_tag_dict[(prev_tag, tag)] = self._num_total_features
							self._num_total_features += 1


	def define_f105(self):
		"""
		Extract out of text all one tags
		:param file_path: full path of file to read
			return all one tags with index of appearance
		"""
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if (((tag) not in self._unigram_tag_dict.keys()) and (self._file_statistics.unigram_tags_count_dict[(tag)] >= self._threshold)):
						self._unigram_tag_dict[(tag)] = self._num_total_features
						self._num_total_features += 1

	def define_first_capital_letter_tag_features_indices(self):
		with open(self._file_statistics.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if word[0].isupper():
						if (((word[0], tag) not in self._first_capital_letter_tags_dict.keys()) and (self._file_statistics.capital_letter_tags_count_dict[(word[0], tag)] >= self._threshold)):
							self._first_capital_letter_tags_dict[(word[0], tag)] = self._num_total_features
							self._num_total_features += 1

	def define_company_feature_index(self):
		self._company_feature_index = self._num_total_features
		self._num_total_features += 1

	def define_all_capital_letters_feature_index(self):
		self._all_capital_letters_index = self._num_total_features
		self._num_total_features += 1

	def define_title_feature_index(self):
		self._title_index = self._num_total_features
		self._num_total_features += 1

	def define_common_adj_suffix_feature_index(self):
		self._common_adj_suffix_index = self._num_total_features
		self._num_total_features += 1

	def define_plural_feature_index(self):
		self._plural_index = self._num_total_features
		self._num_total_features += 1

	def define_plural_and_capital_feature_index(self):
		self._plural_and_capital_index = self._num_total_features
		self._num_total_features += 1

	def define_is_number_feature_index(self):
		self._number_index = self._num_total_features
		self._num_total_features += 1

	def create_history_dict(self):
		history_sequences = self.create_history_sequences_of_length_4(self._file_statistics.file_path)
		history_dict = {}
		index_history = 0
		for history in history_sequences:
			if history not in history_dict.keys():
				history_dict[history] = index_history
				index_history += 1
		self._histories_dict = history_dict

	def create_features_sparse_matrix(self,histories_dict):
		features_list_col = []
		row_list = []
		for i, history in enumerate(histories_dict.keys()):
			features_list = self.represent_input_with_features(history)
			features_list_col.extend(features_list)
			row_list.extend([i] * len(features_list))
		data_list = [1] * len(features_list_col)  # instead of 1 maybe add the count number of the feature
		sparse_features_matrix = csr_matrix((data_list, (row_list, features_list_col)),
											shape=(len(histories_dict.keys()), self._num_total_features))
		return sparse_features_matrix

	def represent_input_with_features(self,history):
		"""
			Extract feature vector in per a given history
			:param history: touple{word, pptag, ptag, ctag, nword, pword}
			:param word_tags_dict: word\tag dict
				Return a list with all features that are relevant to the given history
		"""
		try:
			prev_x2_word, prev_x2_tag, prev_word, prev_tag, curr_word, curr_tag, next_word, next_tag = history
		except:
			prev_x2_word, prev_x2_tag, prev_word, prev_tag, curr_word, curr_tag, next_tag = history
		features = []

		word_tags_dict = self._words_tags_dict
		if (curr_word, curr_tag) in word_tags_dict.keys():
			features.append(word_tags_dict[(curr_word, curr_tag)])

		suffixes_tags_dict = self._suffixes_tags_dict
		for suffix_length in range(1, 5):
			if suffix_length == len(curr_word):
				break
			curr_suffix = curr_word[-suffix_length:len(curr_word)]
			if (curr_suffix, curr_tag) in suffixes_tags_dict.keys():
				features.append(suffixes_tags_dict[(curr_suffix, curr_tag)])

		prefixes_tags_dict = self._prefixes_tags_dict
		for prefixes_length in range(1, 5):
			if prefixes_length == len(curr_word):
				break
			curr_prefixes = curr_word[0:prefixes_length]
			if (curr_prefixes, curr_tag) in prefixes_tags_dict.keys():
				features.append(prefixes_tags_dict[(curr_prefixes, curr_tag)])

		trigram_tag_dict = self._trigram_tag_dict
		if (prev_x2_tag, prev_tag, curr_tag) in trigram_tag_dict.keys():
			features.append(trigram_tag_dict[(prev_x2_tag, prev_tag, curr_tag)])

		bigram_tag_dict = self._bigram_tag_dict
		if (prev_tag, curr_tag) in bigram_tag_dict.keys():
			features.append(bigram_tag_dict[(prev_tag, curr_tag)])

		unigram_tag_dict = self._unigram_tag_dict
		if (curr_tag) in unigram_tag_dict.keys():
			features.append(unigram_tag_dict[(curr_tag)])

		first_capital_letter_tags_dict = self._first_capital_letter_tags_dict
		if (curr_word[0], curr_tag) in first_capital_letter_tags_dict.keys():
			features.append(first_capital_letter_tags_dict[(curr_word[0], curr_tag)])

		capital_letter_word_tags_dict = self._capital_letter_word_tags_dict
		if (curr_word, curr_tag) in capital_letter_word_tags_dict.keys():
			features.append(capital_letter_word_tags_dict[(curr_word, curr_tag)])

		if curr_word.isupper():
			features.append(self._all_capital_letters_index)

		if bool(re.search('^(Ltd|Ltd.|S.A.|SA|A.G.|AG|N.V.|NV|Ltee|B.V|BV|GmbH|L.L.C|LLC|SIA|Sia|Inc.|Inc|Corp.|Corp|Pte.)$',curr_word)):
			features.append(self._company_feature_index)

		try:
			float(curr_word)
			if curr_tag == 'CD':
				features.append(self._number_index)
		except ValueError:
			pass

		if bool(re.search('^(Mr.|Mrs.|Ms.|Miss|Madam|Aunt|Uncle|Dr.|Prof.|Doc.)$',curr_word)):
			features.append(self._title_index)

		if (((curr_word[-3:len(curr_word)] in ['ial', 'ian', 'ary', 'ive', 'ish', 'ous', 'ose', 'ant', 'ent', 'ile']) or
			 (curr_word[-2:len(curr_word)] in ['al', 'an', 'ic']) or (
					 curr_word[-4:len(curr_word)] in ['able', 'ible', 'full', 'less', 'like', ])) and (
				curr_tag == 'JJ')):
			features.append(self._common_adj_suffix_index)

		if curr_word[-1] == 's' and curr_tag == 'NNS':
			features.append(self._plural_index)

		if curr_word[-1] == 's' and curr_word[0].isupper() and curr_tag == 'NNPS':
			features.append(self._plural_and_capital_index)

		return features

	def create_history_all_pos_tags(self):
		histories_list_all_pos_tags = []
		for history in self._histories_dict.keys():
			history_all_pos_tags = {}
			for pos_tag in self._file_statistics.pos_tags:
				temp = list(history)
				temp[5] = pos_tag
				temp = tuple(temp)
				history_all_pos_tags[temp] = self._histories_dict[history]
			histories_list_all_pos_tags.append(history_all_pos_tags)
		self._all_tag_histories_list = histories_list_all_pos_tags

	@staticmethod
	def create_history_sequences_of_length_4(file_path):
		list_history = []
		with open(file_path) as f:
			for line in f:
				splited_words = line.split()
				# del splited_words[-1]
				if (len(splited_words) > 3):
					list_history.append(tuple(
						['*', '*', '*', '*'] + [token for i in range(2) for token in splited_words[i].split("_")]))
					list_history.append(
						tuple(['*', '*'] + [token for i in range(3) for token in splited_words[i].split("_")]))
					window(splited_words, 4, list_history)
					list_history.append(tuple(
						[token for i in reversed(range(1, 4)) for token in splited_words[-i].split("_")] + ['*STOP*']))
				elif (len(splited_words) == 1):
					list_history.append(
						tuple(['*', '*', '*', '*'] + [token for token in splited_words[0].split("_")] + ['*STOP*']))
				elif (len(splited_words) == 2):
					list_history.append(tuple(
						['*', '*', '*', '*'] + [token for i in range(2) for token in splited_words[i].split("_")]))
					list_history.append(tuple(
						['*', '*'] + [token for i in range(2) for token in splited_words[i].split("_")] + ['*STOP*']))
				elif (len(splited_words) == 3):
					list_history.append(tuple(
						['*', '*', '*', '*'] + [token for i in range(2) for token in splited_words[i].split("_")]))
					list_history.append(
						tuple(['*', '*'] + [token for i in range(3) for token in splited_words[i].split("_")]))
					list_history.append(
						tuple([token for i in range(3) for token in splited_words[i].split("_")] + ['*STOP*']))
		return (list_history)

def window(line, n, list_history):
	it = iter(line)
	temp = tuple(islice(it, n))
	res = [x.split("_") for x in temp]
	res = tuple([token for pair in res for token in pair])
	if len(temp) == n:
		list_history.append(res)
	for elem in it:
		temp = temp[1:] + (elem,)
		res = [x.split("_") for x in temp]
		res = tuple([token for pair in res for token in pair])
		list_history.append(res)