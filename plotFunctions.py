import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from wordcloud import WordCloud

def surface(matrix):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(matrix.shape[0], matrix.shape[1], matrix)
	fig.show()

#def createWordCloud(wordList):
#    wordcloud = WordCloud(max_font_size=100)
#    wordcloud.fit_words(wordList)
#    plt.imshow(wordcloud)
#    plt.axis('off')
#    plt.show()


