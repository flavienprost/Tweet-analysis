#This file takes a "testing_features.p" file and anlyzes how the testing sentences are described by
#a certain set of features

#This function aims at understanding the right N_features parameter


N_feat=12000


import cPickle as pickle


testing_data= pickle.load(open('testing_features_N_feat='+ str(N_feat)+ '.p', "rb" ) )
#testing_data=testing_data.todense()
sum=[]
Number_of_zero=0
Number_of_ones=0

print('Afficher')
print(len(testing_data))
print(len(testing_data[1]))
print(testing_data[1])

i=0
while i <len(testing_data) and i<358:
	i=i+1
	current_sum=0
	for j in range(0,N_feat):
		current_sum=current_sum+testing_data[i][j]
	sum.append(current_sum)
	if current_sum==0: Number_of_zero=Number_of_zero+1
	if current_sum==1: Number_of_ones=Number_of_ones+1


print('Number of zeros is %s'%Number_of_zero)
print('Number of ones is %s'%Number_of_ones)
print('Number of ones and zero is %s'%(Number_of_ones+Number_of_zero))
print('Number of testing data is %s'%len(testing_data))


#N=3000:
#Number of zeros is 272
#Number of ones is 161
#Number of ones and zero is 433

#N=6000
#Number of zeros is 121
#Number of ones is 170
#Number of ones and zero is 291
#Number of testing data is 498

#N=10000
#Number of zeros is 40
#Number of ones is 91
#Number of ones and zero is 131
#Number of testing data is 498

