import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil


shutil.rmtree('matplob/', ignore_errors=True)
os.makedirs('matplob/', exist_ok=True)
os.makedirs('matplob/raw', exist_ok=True)
os.makedirs('matplob/label', exist_ok=True)


def draw_label(index, shape_index, location, shape, color, markersize, mew):
    root = 'matplob/label/' + str(index) + "/"
    os.makedirs(root, exist_ok=True)
    plt.figure(figsize=(2.24, 2.24))
    plt.axis(xmax=0, xmin=15)
    plt.axis(ymax=0, ymin=15)
    plt.axis('off')
    plt.plot(
        location[0],
        location[1],
        shape,
        color=color,
        markersize=markersize,
        mew=mew)
    plt.savefig(root + str(shape_index) + ".jpg")
    plt.close()


def create_dataset(n_sample=7500):
  """Creates toy dataset and save to disk."""
  concept = np.reshape(np.random.randint(2, size=15 * n_sample),
                       (-1, 15)).astype(np.bool_)
  print(concept.shape)
  # concept[:15, :15] = np.eye(15)
  location = [(1.3, 1.3), (4.3, 1.3), (7.3, 1.3), (10.3, 1.3), (13.3, 1.3),
              (1.3, 4.3), (4.3, 4.3), (7.3, 4.3), (10.3, 4.3), (13.3, 4.3),
              (1.3, 7.3), (4.3, 7.3), (7.3, 7.3), (10.3, 7.3), (13.3, 7.3),
              (1.3, 10.3), (4.3, 10.3), (7.3, 10.3), (10.3, 10.3), (13.3, 10.3),
              (1.3, 13.3), (4.3, 13.3), (7.3, 13.3), (10.3, 13.3), (13.3, 13.3)]
  location_bool = np.zeros(25)

  # x = np.zeros((n_sample, width, height, 3))

  color_array = ['green', 'red', 'blue', 'black', 'orange', 'purple', 'yellow']

  for i in range(n_sample):
    plt.figure(figsize=(2.24, 2.24))
    plt.axis(xmax=0, xmin=15)
    plt.axis(ymax=0, ymin=15)
    plt.axis('off')

    location_bool = np.zeros(25)
    if i % 100 == 0:
      print('{} images are created'.format(i))
    if concept[i, 5] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          'x',
          color=color_array[np.random.randint(100) % 7],
          markersize=13,
          mew=4)
    if concept[i, 6] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          '3',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 7] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          's',
          color=color_array[np.random.randint(100) % 7],
          markersize=15,
          mew=4)
    if concept[i, 8] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          'p',
          color=color_array[np.random.randint(100) % 7],
          markersize=15,
          mew=4)
    if concept[i, 9] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          '_',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 10] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          'd',
          color=color_array[np.random.randint(100) % 7],
          markersize=12,
          mew=4)
    if concept[i, 11] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          'D',
          color=color_array[np.random.randint(100) % 7],
          markersize=12,
          mew=4)
    if concept[i, 12] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          "v",
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 13] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          'o',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 14] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      plt.plot(
          location[a][0],
          location[a][1],
          '.',
          color=color_array[np.random.randint(100) % 7],
          markersize=20,
          mew=4)
    if concept[i, 0] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      color = color_array[np.random.randint(100) % 7]
      plt.plot(
          location[a][0],
          location[a][1],
          '+',
          color=color,
          markersize=20,
          mew=4)
      draw_label(i, 0, location[a], '+', color, 20, 4)
    if concept[i, 1] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      color = color_array[np.random.randint(100) % 7]
      plt.plot(
          location[a][0],
          location[a][1],
          '1',
          color=color,
          markersize=20,
          mew=4)
      draw_label(i, 1, location[a], '1', color, 20, 4)
    if concept[i, 2] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      color = color_array[np.random.randint(100) % 7]
      plt.plot(
          location[a][0],
          location[a][1],
          '*',
          color=color,
          markersize=15,
          mew=3)
      draw_label(i, 2, location[a], '*', color, 15, 3)
    if concept[i, 3] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      color = color_array[np.random.randint(100) % 7]
      plt.plot(
          location[a][0],
          location[a][1],
          '<',
          color=color,
          markersize=13,
          mew=4)
      draw_label(i, 3, location[a], '<', color, 13, 4)
    if concept[i, 4] == 1:
      a = np.random.randint(25)
      while location_bool[a] == 1:
        a = np.random.randint(25)
      location_bool[a] = 1
      color = color_array[np.random.randint(100) % 7]
      plt.plot(
          location[a][0],
          location[a][1],
          'h',
          color=color,
          markersize=15,
          mew=4)
      draw_label(i, 4, location[a], 'h', color, 15, 4)

    plt.savefig('matplob/raw/' + str(i) + '.jpg')
    plt.close()

  # create label by booling functions
  y = np.zeros((n_sample, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])

  # np.save('x_data.npy', x)
  np.save('y_data.npy', y)
  np.save('concept_data.npy', concept)


create_dataset(n_sample=20000)