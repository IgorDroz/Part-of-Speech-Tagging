class file_statistics():

	def __init__(self,file_path):
		self.file_path = file_path
		self.num_total_features = 0  # Total number of features accumulated
		self.num_words = 0
		self.pos_tags = {}

		# Init all features dictionaries
		self.words_tags_count_dict = {}
		self.suffixes_tags_count_dict = {}
		self.prefixes_tags_count_dict = {}
		self.trigram_tags_count_dict = {}
		self.bigram_tags_count_dict = {}
		self.unigram_tags_count_dict = {}
		self.capital_letter_tags_count_dict = {}
		self.capital_word_tags_count_dict = {}
		self.next_word_tags_count_dict = {}
		self.prev_word_tags_count_dict = {}

		self.calc_unigram_statistics()
		self.calc_unigram_suffix_statistics()
		self.calc_unigram_prefix_statistics()
		self.calc_trigram_tag_statistics()
		self.calc_bigram_tag_statistics()
		self.calc_unigram_tag_statistics()
		self.calc_first_capital_letter_tag_statistics()
		self.calc_capital_letter_word_tag_statistics()
		self.calc_next_word_tag_statistics()
		self.calc_prev_word_tag_statistics()

	def calc_unigram_statistics(self):
		"""
			Extract out of text all statistics about unigrams

		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if (word, tag) not in self.words_tags_count_dict.keys():
						self.num_total_features += 1
						if tag not in self.pos_tags.keys():
							self.pos_tags[tag] = 1
						else:
							self.pos_tags[tag] += 1
						self.words_tags_count_dict[(word, tag)] = 1
					else:
						self.words_tags_count_dict[(word, tag)] += 1

	def calc_next_word_tag_statistics(self):
		"""
			Extract out of text all word/tag pairs
			:param file_path: full path of the file to read
				return all word/tag pairs with index of appearance
		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				#del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx == (len(splited_words) - 1):
						next_word = '*STOP*'
					else:
						next_word = splited_words[idx + 1]
					if (next_word, tag) not in self.next_word_tags_count_dict.keys():
						self.num_total_features += 1
						self.next_word_tags_count_dict[(next_word, tag)] = 1
					else:
						self.next_word_tags_count_dict[(next_word, tag)] += 1

	def calc_prev_word_tag_statistics(self):
		"""
			Extract out of text all word/tag pairs
			:param file_path: full path of the file to read
				return all word/tag pairs with index of appearance
		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				#del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx == 0:
						prev_word = '*'
					else:
						prev_word = splited_words[idx - 1]
					if (prev_word, tag) not in self.prev_word_tags_count_dict.keys():
						self.num_total_features += 1
						self.prev_word_tags_count_dict[(prev_word, tag)] = 1
					else:
						self.prev_word_tags_count_dict[(prev_word, tag)] += 1


	def calc_unigram_suffix_statistics(self):
		"""

		calc statistics about suffixes of unigrams , (suffixes with length of smaller than 4)

		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					for suffix_length in range(1,5):
						if suffix_length==len(word):
							break
						suffix = word[-suffix_length:len(word)]
						if (suffix, tag) not in self.suffixes_tags_count_dict.keys():
							self.num_total_features += 1
							self.suffixes_tags_count_dict[(suffix, tag)] = 1
						else:
							self.suffixes_tags_count_dict[(suffix, tag)] += 1

	def calc_unigram_prefix_statistics(self):
		"""
		calc statistics about prefixes of unigrams , (prefixes with length of smaller than 4)

		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					for prefix_length in range(1,5):
						if prefix_length==len(word):
							break
						prefix = word[0:prefix_length]
						if (prefix, tag) not in self.prefixes_tags_count_dict.keys():
							self.num_total_features += 1
							self.prefixes_tags_count_dict[(prefix, tag)] = 1
						else:
							self.prefixes_tags_count_dict[(prefix, tag)] += 1

	def calc_trigram_tag_statistics(self):
		"""
		Calculate trigram tag statistics

		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx > 1:
						prev_word, prev_tag = splited_words[idx - 1].split('_')
						prev_x2_word, prev_x2_tag = splited_words[idx - 2].split('_')
						if (prev_x2_tag, prev_tag, tag) not in self.trigram_tags_count_dict.keys():
							self.num_total_features += 1
							self.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] = 1
						else:
							self.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] += 1
					elif idx == 1:
						prev_word, prev_tag = splited_words[idx - 1].split('_')
						prev_x2_word, prev_x2_tag = '*', '*'
						if (prev_x2_tag, prev_tag, tag) not in self.trigram_tags_count_dict.keys():
							self.num_total_features += 1
							self.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] = 1
						else:
							self.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] += 1
					elif idx == 0:
						prev_word, prev_tag = '*','*'
						prev_x2_word, prev_x2_tag = '*', '*'
						if (prev_x2_tag, prev_tag, tag) not in self.trigram_tags_count_dict.keys():
							self.num_total_features += 1
							self.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] = 1
						else:
							self.trigram_tags_count_dict[(prev_x2_tag, prev_tag, tag)] += 1


	def calc_bigram_tag_statistics(self):
		"""
		Calculate bigram tag statistics

		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if idx > 0:
						prev_word, prev_tag = splited_words[idx - 1].split('_')
						if (prev_tag, tag) not in self.bigram_tags_count_dict.keys():
							self.num_total_features += 1
							self.bigram_tags_count_dict[(prev_tag, tag)] = 1
						else:
							self.bigram_tags_count_dict[(prev_tag, tag)] += 1
					else:
						prev_word, prev_tag = '*', '*'
						if (prev_tag, tag) not in self.bigram_tags_count_dict.keys():
							self.num_total_features += 1
							self.bigram_tags_count_dict[(prev_tag, tag)] = 1
						else:
							self.bigram_tags_count_dict[(prev_tag, tag)] += 1


	def calc_unigram_tag_statistics(self):
		"""
		Calculate unigram tag statistics

		"""
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if (tag) not in self.unigram_tags_count_dict.keys():
						self.num_total_features += 1
						self.unigram_tags_count_dict[(tag)] = 1
					else:
						self.unigram_tags_count_dict[(tag)] += 1


	def calc_first_capital_letter_tag_statistics(self):
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if word[0].isupper():
						if (word[0], tag) not in self.capital_letter_tags_count_dict.keys():
							self.num_total_features += 1
							self.capital_letter_tags_count_dict[(word[0], tag)] = 1
						else:
							self.capital_letter_tags_count_dict[(word[0], tag)] += 1

	def calc_capital_letter_word_tag_statistics(self):
		with open(self.file_path) as f:
			for line in f:
				splited_words = line.split()
				del splited_words[-1]
				for idx in range(len(splited_words)):
					word, tag = splited_words[idx].split('_')
					if sum([1 for letter in word if letter.isupper()])>0:
						if (word, tag) not in self.capital_word_tags_count_dict.keys():
							self.num_total_features += 1
							self.capital_word_tags_count_dict[(word, tag)] = 1
						else:
							self.capital_word_tags_count_dict[(word, tag)] += 1
