import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.ticker as ticker
from matplotlib import image
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

import seaborn as sns

def resize(np_img,basewidth=200):
    img = Image.fromarray(np_img)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return np.array(img)

def hexify(palette):
    return['#%s' % ''.join(('%02x' % round(p) for p in colour)) for colour in palette]

def KMeansModel(imgfile=None,n_clusters=5):
    # Loading and resizing the selected image. It is necessary to reshape the numpy array to train our model.
    img = image.imread(imgfile)
    img = resize(img,200)
    img_arr = img.reshape((img.shape[0]*img.shape[1],3))
    
    cluster = KMeans(n_clusters=n_clusters,init='random',n_init=10, max_iter=300,random_state=0)
    cluster.fit_predict(img_arr)
    
    return cluster, img

def plot_results(cluster, img):
    # We obtain the dominant colours given by cluster.cluster_centers_. We also generate a colour labels array to be used in our scatter plot.
    colours = np.int_(cluster.cluster_centers_.round())
    colour_labels = hexify(colours)
    h = hexify(colours[cluster.labels_])
    
    fig = plt.figure(figsize=(20,10))
    gs = grd.GridSpec(3, 2, width_ratios=[2,1])
    sns.set_style('whitegrid')
    
    # Scatter plot
    img_arr = img_arr = img.reshape((img.shape[0]*img.shape[1],3))
    df = pd.DataFrame(img_arr,columns=['R','G','B'])
    ax = fig.add_subplot(gs[:, 0],projection='3d')
    ax.scatter(df['R'],df['G'],df['B'],c=h)
    ax.view_init(30, -120)
    
    # Image plot
    ax1 = fig.add_subplot(gs[0:2,1])
    ax1.imshow(img)
    plt.grid(False)
    
    # Colour palette
    ax2 = fig.add_subplot(gs[2,1])
    x = np.arange(len(colour_labels))
    y = np.ones(len(x))
    ax2.bar(x,y,width=1.0,color=colours/256)
    ax2.set_xticklabels(['']+colour_labels,rotation=30)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.get_yaxis().set_ticks([])

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.grid(False)

    plt.show()
    return colour_labels

def colormap(cluster,img):
    colours = np.int_(cluster.cluster_centers_.round())
    cmap = colours[cluster.labels_]
    cmap = cmap.reshape((img.shape[0],img.shape[1],3))
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cmap)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)