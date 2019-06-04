# script to compute the local semantic change of words
# the main function is local_measure(word, knn, bins)

# INPUT
# word  string    is the word that you want to analyze
# knn   int       is the number of nearest neighbours that you want to consider
# bins  list      is a series of numbers used to store the sequence of time periods


# OUTPUT
# The output will be in tsv format
# new_words_        words that are in t+1 but not in t
# lost_words_       words that are in t but not in t+1
# changes_          words that are in t and t+1 and their differences
# Local_Similarity_ matrix in which are stored the overall differences for each pair of bins


from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_distances
import csv

#### ---- FILES PATH ---- ####
### probably you need to change them

# where the embeddings are located
base_all = 'emb/'

# where the results are saved
base = 'local/'

#### ---- EMBEDDING PARAMETERS ---- ####

### probably you need to change them
### used to construct the file name of the embeddings and to load them

win = 5 # window size
size = 300 # vector dimensions
min_count = 25 # minimal number of occurrences
name = 'WV' # word2vec skip-gram

# ----------------

# time bins used to create the file name of the corresponding embeddings and to load them
bins = [1789, 1830, 1841, 1848, 1855, 1861, 1866, 1870, 1874, 1877, 1880, 1883, 1886, 1889, 1891, 1893, 1895, 1897, 1899, 1901, 1903, 1905, 1907, 1909, 1911, 1913, 1915]

def writeMat(base, name, Mat):
#    Mat = Mat.todense()
    with open(base + name + '.tsv', 'w') as f:
        w = csv.writer(f, delimiter='\t')
        for row in Mat:
            w.writerow(row)

def getSim(embed,word,knn):
    sim = []
    for k in knn:
        if k in embed.vocab:
            s = embed.similarity(word,k)
        else:
            s = 0
        sim.append(s)
    return sim

def comp_changes(knn_t,knn_t1):
    t_words = [w[0] for w in knn_t]
    t1_words = [w[0] for w in knn_t1]
    new_words = [(w,[w1[1] for w1 in knn_t1 if w1[0] == w][0]) for w in t1_words if w not in t_words]
    lost_words = [(w,[w1[1] for w1 in knn_t if w1[0] == w][0]) for w in t_words if w not in t1_words]
    same_words = [w for w in t_words if w in t1_words]
    diffs = [(w,abs([w1[1] for w1 in knn_t if w1[0] == w][0] - [w1[1] for w1 in knn_t1 if w1[0] == w][0])) for w in same_words]

    new_words.sort(key=lambda tup: tup[1], reverse=True)
    lost_words.sort(key=lambda tup: tup[1], reverse=True)
    diffs.sort(key=lambda tup: tup[1], reverse=True)

    return new_words, lost_words, diffs

def local_measure(word,knn,bins):
    # the matrix of size len(bins) x len(bins)
    # where the cosine distance of each pair of bins is computed
    S = []
    print(word, knn)
    for xx in range(len(bins) - 2):
        bin1 = bins[xx]
        bin2 = bins[xx+1]
        time1 = str(bin1)
        time2 = str(bin2)
        # path to the embeddings
        path = base_all + '3EMB-' + name + '-' + 'win_' + str(win) + '-size_' + str(
            size) + '-min_count_' + str(min_count) + '-iter_' + str(time1) + '-' + str(time2)
        # load the embeddings at time t0 using gensim KeyedVectors
        embed = KeyedVectors.load(path)
        # check if the word is in the embedding wocabulary
        if word not in embed.wv.vocab:
            print(word + ' not in base_embed\'s vocabulary')
            continue
        else:
            knn_t = embed.most_similar(word, topn=knn)
            knn_t_words = [k[0] for k in knn_t]
            knn_t_sims = [k[1] for k in knn_t]
        if xx > 0:
            S.append([0]*xx)
        else:
            S.append([])
            knn_t0 = knn_t
            knn_t0_words = knn_t_words
            time0 = str(bin1) + '_' + str(bin2)
        # only the values of S above the main diagonal are non-zero
        # this because cosine distance is simmetric and the values on the diagonal are 0
        for x in range(xx+1,len(bins)-1):
            time11 = str(bins[x])
            time22 = str(bins[x+1])
            time = time1 + '-' + time2 + '_' + time11 + '-' + time22
            time00 = time0 + '_' + time11 + '-' + time22
            print(time)
            path_t1 = base_all + '3EMB-' + name + '-' + 'win_' + str(win) + '-size_' + str(size) + '-min_count_' + str(min_count) + '-iter_' + time11 + '-' + time22

            # load the embeddings at time t1 using gensim KeyedVectors
            embed_t1 = KeyedVectors.load(path_t1)

            if word not in embed_t1.wv.vocab:
                print(word + ' not in embed\'s vocabulary')
                continue
            else:
                knn_t1 = embed_t1.most_similar(word, topn=knn)
                knn_t1_words = [k[0] for k in knn_t1]
                knn_t1_sims = [k[1] for k in knn_t1]
            # create the second order vector as in:
            # Hamilton, William L., Jure Leskovec, and Dan Jurafsky. "Cultural shift or linguistic drift? comparing two computational measures of semantic change." Proceedings of the Conference on Empirical Methods in Natural Language Processing. Conference on Empirical Methods in Natural Language Processing. Vol. 2016. NIH Public Access, 2016.
            # Equation 2
            s_t = getSim(embed,word,knn_t_words+knn_t1_words)
            s_t1 = getSim(embed_t1,word,knn_t_words+knn_t1_words)
            dist = cosine_distances([s_t,s_t1]).tolist()[0][1]

            new_words, lost_words, diffs = comp_changes(knn_t,knn_t1)
            new_words0, lost_words0, diffs0 = comp_changes(knn_t0,knn_t1)

            writeMat(base, 'new_words_' + time + '_' + word + '_' + str(knn), new_words)
            writeMat(base, 'lost_words_' + time + '_' + word + '_' + str(knn), lost_words)
            writeMat(base, 'changes_' + time + '_' + word + '_' + str(knn), diffs)

            writeMat(base, 'new_words_' + time00 + '_' + word + '_' + str(knn), new_words0)
            writeMat(base, 'lost_words_' + time00 + '_' + word + '_' + str(knn), lost_words0)
            writeMat(base, 'changes_' + time00 + '_' + word + '_' + str(knn), diffs0)

            S[xx].append(dist)
    writeMat(base, 'Local_Similarity_' + name + '_' + word + '_' + str(knn), S)


# EXAMPLE
local_measure('juif',100,bins)