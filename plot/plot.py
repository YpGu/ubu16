import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


core = {}

'''
  scatters of 1d ideal points for media and politician accounts
'''
def plot_with_annotations(filedir):
  id2v = {}; id2party = {}
  fin = open(filedir)
  lines = fin.readlines()
  for line in lines:
    ls = line.split('\t')
    uid = ls[0]
    clas = ls[1]
    tname = ls[2]
    v = float(ls[3])

    id2v[tname] = v
    if clas == 'L':
      id2party[tname] = 'b'
    elif clas == 'C':
      id2party[tname] = 'r'
    elif clas == 'N':
      id2party[tname] = 'k'

  fin.close()

  fig = plt.figure(figsize = (6,11))
#  ax = fig.add_axes([0, 0.1, 1, 1])
  gcf = plt.gcf()
  text_height = -750
  x = -300
  x_offset = 0
  nota_size = 10
  color_dict = {'r':'red', 'b':'blue', 'k':'black'}
#  dir_dict = {'r':'left', 'b':'right', 'k':'left'}
  dir_dict = {'r':'right', 'b':'right', 'k':'right'}
#  dir2_dict = {'r':'bottom', 'b':'top', 'k':'top'}
  dir2_dict = {'r':'top', 'b':'top', 'k':'top'}
#  for tname in id2party:
  for t in sorted(id2v.items(), key = lambda xx: xx[1], reverse = False):
    tname = t[0]
    y = id2v[tname]
    party = id2party[tname]
    plt.scatter(y, x,  marker = 'o', c = party, s = 50, cmap = plt.get_cmap('Spectral'))
    plt.annotate(tname, xy = (-2, x), xytext = (0, 0), textcoords = 'offset points', 
	  ha = dir_dict[party], va = dir2_dict[party], 
	  fontweight = 100, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5)
#	  arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 1)
#	  arrowprops = dict(arrowstyle = '-', connectionstyle = 'bar', facecolor = 'black', linewidth = 1)
	  )
    text_height += 50
    x += 20
#    if tname == 'WSJ' or tname == 'CNBC' or tname == 'FoxNews':
#      x += 2000

    '''
    if tname == 'CNBC':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, -200), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'usnews':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, 450), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'Newsweek':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, 400), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'CNN':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, 300), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'YahooNews':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, 400), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'ABC':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, 150), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'cspan':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, -100), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'WSJ':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, -600), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    elif tname == 'washingtonpost':
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, 250), textcoords = 'offset points', ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    else:
      plt.annotate(tname, xy = (x, 0), xytext = (x_offset, text_height), textcoords = 'offset points', 
	    ha = dir_dict[party], va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0.5),
	    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 3))
    '''

  frame = plt.gca()
#  frame.axes.get_yaxis().set_visible(False)
  frame.axes.get_yaxis().set_ticks([])
  plt.xlabel('Ideology', fontsize = 20)
  plt.ylabel('Twitter screen name', fontsize = 20)
  plt.xticks([-2+x for x in range(5)], fontsize = 16)
  plt.xlim([-3.8,2.5])
  plt.ylim([-325,270])
  plt.show()

#  gcf.set_size_inches(36,28)
#  gcf.savefig('./save/1dideology.png')

'''
  histogram for Twitter users
'''
def read_core():
  fin = open('./data/core_users')
  lines = fin.readlines()
  for l in lines:
    core[int(l.split('\t')[1])] = 0
  fin.close()

def to_percent(y, pos):
  s = str(100*y)
  return s + '%'

def plot_histogram(filedir):
  read_core()
  vs = []

  fin = open(filedir)
  lines = fin.readlines()
  for line in lines:
    ls = line.split('\t')
    uid = int(ls[0]); v = -float(ls[1])
    if uid in core:
      vs.append(v)

  # the histogram of the data
  plt.figure(figsize = (25,8))
#  plt.subplot(2,1,1)
  hist, bin_edges = np.histogram(vs, bins = [-3 + 0.04 * x for x in range(151)])
  hist_norm = [i/float(len(vs)) for i in hist]
  plt.bar(bin_edges[:-1], hist_norm, width = 0.05, color = 'g')
  plt.xlim(-2.5,2.5)

  plt.xlabel('Ideology', fontsize = 30)
  plt.xticks([-2+x for x in range(5)], fontsize = 25)
#  plt.yticks([200 * x for x in range(9)], fontsize = 25)
  plt.yticks([i/100.0 for i in range(7)])
  formatter = FuncFormatter(to_percent)
  plt.gca().yaxis.set_major_formatter(formatter)
  plt.ylabel('Percentage of Users', fontsize = 25)
  plt.tick_params(axis='y', labelsize = 25)

  plt.savefig('./save/core_users_dist.png')
  plt.show()

  '''
  peripheral users
  '''

  vs = []
  fin = open(filedir)
  lines = fin.readlines()
  for line in lines:
    ls = line.split('\t')
    uid = int(ls[0]); v = -float(ls[1])
    if uid not in core:
      vs.append(v)

  # the histogram of the data
  plt.figure(figsize = (25,8))
#  plt.subplot(2,1,2)
  hist, bin_edges = np.histogram(vs, bins = [-3 + 0.04 * x for x in range(151)])
  hist_norm = [i/float(len(vs)) for i in hist]
  plt.bar(bin_edges[:-1], hist_norm, width = 0.05, color = 'g')
  plt.xlim(-2.5,2.5)

  plt.xlabel('Ideology', fontsize = 30)
  plt.xticks([-2+x for x in range(5)], fontsize = 25)
  plt.ylabel('Percentage of Users', fontsize = 25)
#  plt.title('Peripheral users', fontsize = 40)
#  plt.yticks([200 * x for x in range(5)], fontsize = 25)
  formatter = FuncFormatter(to_percent)
  plt.gca().yaxis.set_major_formatter(formatter)
  plt.tick_params(axis='y', labelsize = 25)

  plt.savefig('./save/pher_users_dist.png')
  plt.show()

def cold_start(filedir):
  auto = []; fixed = []; single = []; base = []
  xs = [x for x in range(8)]
  fin = open(filedir)
  lines = fin.readlines()
  for line in lines:
    ls = line.split('\t')
    auto.append(float(ls[0]))
    fixed.append(float(ls[1]))
    single.append(float(ls[2]))
    base.append(float(ls[3]))
  fin.close()

  fig = plt.figure()
#  plt.scatter(xs, auto, c = ['r' for u in auto])
#  plt.scatter(xs, fixed, c = ['g' for u in fixed])
  plt.scatter(xs, single, c = ['b' for u in single])
#  plt.scatter(xs, base, c = ['k' for u in base])
  line1, = plt.plot(xs, auto, 'ro-')
  line2, = plt.plot(xs, fixed, 'go--')
  line3, = plt.plot(xs, single, 'bo:')
  line4, = plt.plot(xs, base, 'ko-.')
  plt.legend([line1, line2, line3, line4], ['Automatic', 'Uniform (fixed)', 'Single Network', 'BIPM'], loc = 4)
  plt.xlim(-0.5,7.5)
  plt.xlabel('Threshold (degree in the training dataset)', fontsize = 16)
  plt.ylabel('Link prediction AUC', fontsize = 16)

  plt.savefig('./save/' + filedir.split('/')[-1].split('.')[0] + '.png')
  plt.show()


if __name__ == '__main__':
  f = sys.argv[1]

#  plot_with_annotations(f)	# Usage: python plot.py ./data/media (gu2)
  plot_histogram(sys.argv[1])
#  cold_start(f)

