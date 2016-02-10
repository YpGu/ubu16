import matplotlib.pyplot as plt

# plot the average idp of R/D's in each congress 
def plotidp(option):
	xs_auto = []; xs_fix = []; xs_single = []; xs_bline = []; xs_ave = []

	fdir = ''
	outdir = './save/'
	if option == 1:
		fdir = './data/mention_cs.txt'
		outdir = './save/cold_start_mention.png'
	elif option == 2:
		fdir = './data/retweet_cs.txt'
		outdir = './save/cold_start_retweet.png'
	fin = open(fdir)
	lines = fin.readlines()
	for line in lines:
		ls = line.split('\t')
		xs_auto.append(float(ls[0]))
		xs_fix.append(float(ls[1]))
		xs_single.append(float(ls[2]))
		xs_bline.append(float(ls[3]))
		xs_ave.append(float(ls[4]))
	fin.close()
	ys = range(len(lines))

	fig = plt.figure()
#	plt.scatter(ys, xs_auto, c = ['r' for u in ys])
	l1, = plt.plot(ys, xs_auto, 'ro-', label = 'l1')
#	plt.scatter(ys, xs_fix, c = ['g' for u in ys])
	l2, = plt.plot(ys, xs_fix, 'go--', label = 'l2')
#	plt.scatter(ys, xs_single, c = ['b' for u in ys])
	l3, = plt.plot(ys, xs_single, 'bo:', label = 'l3')
	plt.scatter(ys, xs_bline, c = ['k' for u in ys])
	l4, = plt.plot(ys, xs_bline, 'ko-.', label = 'l4')
	l5, = plt.plot(ys, xs_ave, 'mo--', linewidth = 3, label = 'l5')

	plt.xlabel('Threshold (degree in the training dataset)')
	plt.ylabel('Link Prediction AUC')
	plt.xlim([-0.5,7.5])
#	plt.legend([l1, l2, l3, l4], ['ML-IPM', 'ML-IPM-fixed', 'SL-IPM (mention)', 'B-IPM (mention)'], loc = 4)
	plt.legend([l1, l2, l3, l4, l5], ['ML-IPM', 'ML-IPM-fixed', 'SL-IPM (retweet)', 'B-IPM (retweet)', 'AVER'], loc = 4)
	plt.show()
	fig.savefig(outdir)

if __name__ == '__main__':
	plotidp(1)
	plotidp(2)

