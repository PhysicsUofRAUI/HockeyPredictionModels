import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD
opt = SGD(lr=100)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
my_data = pd.read_csv("NoBlanksAndScoreAsDummy.csv")

score = my_data["score"]

my_data = my_data.drop("score", axis=1)

my_dummies = pd.get_dummies(my_data, prefix=['T1P1', 'T1P2', 'T1P3', 'T1P4', 'T1P5', 'T1P6', 'T1P7', 'T1P8', 'T1P9', 'T1P10', 'T1P11', 'T1P12', 'T1P13', 'T1P14', 'T1P15', 'T1P16', 'T1P17', 'T1P18', 'T1G', 'T2P1', 'T2P2', 'T2P3', 'T2P4', 'T2P5', 'T2P6', 'T2P7', 'T2P8', 'T2P9', 'T2P10', 'T2P11', 'T2P12', 'T2P13', 'T2P14', 'T2P15', 'T2P16', 'T2P17', 'T2P18', 'T2G'])

my_dummies["result"] = score

print(my_dummies)

dataset = my_dummies.values

X = dataset[:,0:12993]
Y = dataset[:,12993]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

print("Number,results,epochs,batch_size,number of layers")
# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(12993, input_dim=12993, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5000, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(5000, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(5000, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=500, batch_size=250, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,3" % (results.mean()*100, results.std()*100))



# # second model
# def create_baseline_one():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_one, epochs=500, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,500,25,1" % (results.mean()*100, results.std()*100))
#
#
#
# # third model
# def create_baseline_two():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_two, epochs=500, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,500,25,2" % (results.mean()*100, results.std()*100))
#
#
# # fourth model
# def create_baseline_three():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_three, epochs=500, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,500,25,4" % (results.mean()*100, results.std()*100))
#
#
#
#
# # fifth model
# def create_baseline_four():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_four, epochs=500, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,500,25,5" % (results.mean()*100, results.std()*100))
#
#
#
#
# # sixth model
# def create_baseline_five():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_five, epochs=500, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,500,25,6" % (results.mean()*100, results.std()*100))
#
#
#
#
#
# # seventh model
# def create_baseline_six():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_six, epochs=500, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,500,25,8" % (results.mean()*100, results.std()*100))
#
#
#
# # eighth model
# def create_baseline_seven():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_seven, epochs=500, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,500,25,10" % (results.mean()*100, results.std()*100))
#
#
#
# #
# #
# #
# #
# #
# #
# #
# # These next models are the same but they have more 1000 epochs
# #
# #
# #
# #
# #
# #
# #
# #
# #
#
# def create_baseline():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,3" % (results.mean()*100, results.std()*100))
#
#
#
# # second model
# def create_baseline_one():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_one, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,1" % (results.mean()*100, results.std()*100))
#
#
#
# # third model
# def create_baseline_two():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_two, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,2" % (results.mean()*100, results.std()*100))
#
#
# # fourth model
# def create_baseline_three():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_three, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,4" % (results.mean()*100, results.std()*100))
#
#
#
#
# # fifth model
# def create_baseline_four():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_four, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,5" % (results.mean()*100, results.std()*100))
#
#
#
#
# # sixth model
# def create_baseline_five():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_five, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,6" % (results.mean()*100, results.std()*100))
#
#
#
#
#
# # seventh model
# def create_baseline_six():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_six, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,8" % (results.mean()*100, results.std()*100))
#
#
#
# # eighth model
# def create_baseline_seven():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_seven, epochs=1000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,1000,25,10" % (results.mean()*100, results.std()*100))
#
#
# #
# #
# #
# #
# #
# # these next ones have 2000 epochs
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
#
# def create_baseline():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,3" % (results.mean()*100, results.std()*100))
#
#
#
# # second model
# def create_baseline_one():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_one, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,1" % (results.mean()*100, results.std()*100))
#
#
#
# # third model
# def create_baseline_two():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_two, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,2" % (results.mean()*100, results.std()*100))
#
#
# # fourth model
# def create_baseline_three():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_three, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,4" % (results.mean()*100, results.std()*100))
#
#
#
#
# # fifth model
# def create_baseline_four():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_four, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,5" % (results.mean()*100, results.std()*100))
#
#
#
#
# # sixth model
# def create_baseline_five():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_five, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,6" % (results.mean()*100, results.std()*100))
#
#
#
#
#
# # seventh model
# def create_baseline_six():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_six, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,8" % (results.mean()*100, results.std()*100))
#
#
#
# # eighth model
# def create_baseline_seven():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
# 	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# 	return model
#
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline_seven, epochs=2000, batch_size=25, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("1, %.2f%% , (%.2f%%) ,2000,25,10" % (results.mean()*100, results.std()*100))
