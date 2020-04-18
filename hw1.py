from file_statistics import file_statistics
from MEMM_predictor import MEMM_predictor

def run_eval(threshold = 2, beam_size=3):
	file_path = 'train1.wtag'
	fs_train1 = file_statistics(file_path)
	MEMM = MEMM_predictor(fs_train1,threshold)

	test_path = 'test1.wtag'
	weights_path = 'weights_train1.pkl'
	MEMM.predict(test_path, weights_path,beam_size,eval_mode=True , plot=True)

def run_train(file_path,threshold = 2,lambda_=0.005,save=True):
	fs_train1 = file_statistics(file_path)
	MEMM = MEMM_predictor(fs_train1,threshold)
	MEMM.fit(lambda_=lambda_,save=save)

def run_inference(train_file_path,test_file_path,beam_size =3 ,threshold = 2):
	fs_train1 = file_statistics(train_file_path)
	MEMM = MEMM_predictor(fs_train1,threshold)
	weights_path = 'weights_'+train_file_path+'.pkl'
	predictions, accuracy = MEMM.predict(test_file_path, weights_path,beam_size)
	return  predictions


if __name__ == '__main__':
	run_eval()
	# run_train(file_path='./train2.wtag')
	# run_inference(train_file_path='./train2.wtag',test_file_path='./comp2.words')





