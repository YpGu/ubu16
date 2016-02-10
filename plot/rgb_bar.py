import sys
import matplotlib.pyplot as plt
import numpy as np

def plot_bar():

  xs = [x for x in range(-255,256)]
  ys = [0 for x in xs]
  cs = []
  for i in range(256):
    cs.append((i/255.0, i/255.0, 1))
  for i in range(1,256):
    cs.append((1, 1-i/255.0, 1-i/255.0))

  '''
  gcf = plt.gcf()
  text_height = -750
  x = -300
  x_offset = 0
  nota_size = 10
  yticks = []
  '''

  fig = plt.figure()
  plt.scatter(xs, ys, c = cs, alpha = 0.1, s = 200, edgecolors='none')
#  plt.scatter([0],[1],c=[(1,0,1)])
#  line1, = plt.plot(xs, auto, 'ro-')
#  line2, = plt.plot(xs, fixed, 'go--')
#  line3, = plt.plot(xs, single, 'bo:')
#  line4, = plt.plot(xs, base, 'ko-.')
#  plt.legend([line1, line2, line3, line4], ['Automatic', 'Uniform (fixed)', 'Single Network', 'BIPM'], loc = 4)
#  plt.xlim(-0.5,7.5)
#  plt.xlabel('Threshold (degree in the training dataset)', fontsize = 16)
#  plt.ylabel('Link prediction AUC', fontsize = 16)

  plt.savefig('./save/rgb_bar.png')
  plt.show()

if __name__ == '__main__':
  plot_bar()

