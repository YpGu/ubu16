import sys
import matplotlib.pyplot as plt
import numpy as np

'''
  scatters of 1d ideal points for media and politician accounts
'''
def plot_with_annotations(filedir):
#  left = []; right = []; neu = []
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

  fig = plt.figure(figsize = (5,11))
#  ax = fig.add_axes([0, 0.1, 1, 1])
  gcf = plt.gcf()
  text_height = -750
  x = -300
  x_offset = 0
  nota_size = 22
  color_dict = {'r':'red', 'b':'blue', 'k':'black'}
#  dir_dict = {'r':'left', 'b':'right', 'k':'left'}
  dir_dict = {'r':'right', 'b':'right', 'k':'right'}
#  dir2_dict = {'r':'bottom', 'b':'top', 'k':'top'}
  dir2_dict = {'r':'top', 'b':'top', 'k':'top'}
#  for tname in id2party:
  yticks = []
  for t in sorted(id2v.items(), key = lambda xx: xx[1], reverse = False):
    tname = t[0]
    yticks.append(tname)
    y = id2v[tname]
    party = id2party[tname]
    if y > 0:
      lr = 'right'
      ys = [i/1000.0 for i in range(int(y*1000))]
    else:
      lr = 'left'
      ys = [-i/1000.0 for i in range(int(-y*1000))]
    xs = [x for y in ys]
    cs = [party for y in ys]
#    plt.scatter(y, x, marker = 'o', c = party, s = 50, cmap = plt.get_cmap('Spectral'))
    plt.scatter(ys, xs, marker = 'o', c = cs, s = 20, cmap = plt.get_cmap('Spectral'), alpha = 0.2, edgecolors='none')
    if y > 0:
      plt.annotate(tname, xy = (-0.1, x + 4), xytext = (0, 0), textcoords = 'offset points', 
  #	  ha = dir_dict[party], va = dir2_dict[party], 
	    ha = lr, va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0)
  #	  arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0', facecolor = 'black', linewidth = 1)
  #	  arrowprops = dict(arrowstyle = '-', connectionstyle = 'bar', facecolor = 'black', linewidth = 1)
	    )
    else:
      plt.annotate(tname, xy = (0.1, x + 4), xytext = (0, 0), textcoords = 'offset points', 
  #	  ha = dir_dict[party], va = dir2_dict[party], 
	    ha = lr, va = dir2_dict[party], 
	    fontweight = 500, fontsize = nota_size, bbox = dict(boxstyle = 'round,pad=0.5', fc = color_dict[party], alpha = 0)
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

  plt.plot([0,0],[-330,-180], 'k--')

  frame = plt.gca()
  frame.axes.get_yaxis().set_visible(False)
  ax = plt.subplot(111)
  ax.spines['right'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['top'].set_visible(False)
  frame.axes.get_yaxis().set_ticks([])
  plt.xlabel('Ideology', fontsize = 20)
#  plt.ylabel('Twitter screen name', fontsize = 20)
  plt.xlim([-1.2,1.2])
#  plt.ylim([-325,300])
  plt.show()

#  gcf.set_size_inches(36,28)
  gcf.savefig('./save/intro-1dideology.png')

'''
  histogram for all Twitter users
'''
def plot_histogram(filedir):
  vs = []
  fin = open(filedir)
  lines = fin.readlines()
  for line in lines:
    ls = line.split('\t')
    uid = ls[0]
    v = float(ls[1])
    vs.append(v)

  # the histogram of the data
  hist, bin_edges = np.histogram(vs, bins = [-3 + 0.06 * x for x in range(101)])
  plt.bar(bin_edges[:-1], hist, width = 0.05, color = 'g')
  plt.xlim(-3,3)

  plt.xlabel('Ideology', fontsize = 20)
  plt.xticks([-3+x for x in range(7)], fontsize = 20)
  plt.ylabel('Number of Users', fontsize = 20)
  plt.yticks([200 * x for x in range(8)], fontsize = 20)

  plt.savefig('all_users_dist.png')
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
  plot_with_annotations(f)	# Usage: python plot.py ./data/gu2
#  plot_histogram(sys.argv[1])
#  cold_start(f)

