import matplotlib.pyplot as plt

def plot_states():
  fin = open('./data/state_colors')
  lines = fin.readlines()
  vs = []
  for l in lines:
    ls = l.split(',')
    vs.append(float(ls[3]) - 0.0024)
  min_v = min(vs)
  max_v = max(vs)
  print min_v, max_v

#  fig = plt.figure(figsize = (6,11))
  for line in lines:
    sID = int(line.split(',')[0])
    sName = line.split(',')[2]
    v = float(line.split(',')[3]) - 0.0024

    if v > 0:
      r = 255
      b = (1.0 - v/max_v) * 255 * 2/3
      g = b
      rgb = (r, g, b)
      print sName, rgb
    elif v < 0:
      b = 255
      r = (1.0 - v/min_v) * 255 * 2/3
      g = r
      rgb = (r, g, b)
      print sName, rgb

#  plt.show()

  fin.close()

if __name__ == '__main__':
  plot_states()

