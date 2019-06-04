from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import csv
import pickle
from numpy import float32 as REAL, array
from gensim import matutils

# DICTIONARIES OF STREAMS

streams_ms = {'religious' : [('spirituel','séculier'),('ange','diable'),('sacré','profane'),('pieux','athée'), ('pieux','païen'), ('pieux','idolâtre'), ('pieux','impie'), ('sacré','maudit'),('vénérable','abject'),('fidèle','infidèle'),('croyant','incroyant'),('religieux','irreligieux'),('dévoué','athée')],
           'economic': [('donner','approprier'),('générosité','cupidité'),('générosités','avidité'),('généreux','avide'),('généreux','avare'),('généreux', 'pingre')],
           'socio-political':[('prodigue','cupide'),('honnête','canaille'),('honneur','honte'),('amical','hostile'),('loyal','fourbe'),('socialiste','capitaliste'),('amusant','parasite'),('amis','ennemi'),('allié','antagoniste'),('conservateur','progressiste'),('autoritaire','révolutionnaire')],
           'racist':[('amusant','parasite'),('normal','étrange'),('supériorité','infériorité'),('égalité','inégalité'),('estimable','laide'),('affable','méchant'),('estimable','infâme'),('sympathie','haine'),('accepté','refusé'),('meilleur','pire'),('national','étranger'),('pur','impur'),('superieure','inférieure'),('pur','infect'),('propre','sale')],
           'conspiratorial':[('loyal','espion'),('honnêteté','trahison'),('loyal','traître'),('clair','mystérieux'),('évident','occulte'),('sincère','trompeur'),('sincère','déloyal'),('bienfaiteur','criminel'),('évident','secret'),('amical','menaçant'),('clair','obscur')],
           'ethic':[('chasteté','érotisme'),('modeste','intrigant'),('décent','indécent'),('vertueux','lascif'),('fidèle','infidèle'),('morale','immorale'),('honnête','malhonnête'),('vertueux','corrompu'),('chaste','dépravé'),('chaste','charnel'),('pur','dépravé')]
           }
#
streams_mp = {'religious' : [('spirituel','séculier'),('anges','diables'),('sacrés','profanes'),('pieux','athées'), ('pieux','païens'), ('pieux','idolâtres'), ('pieux','impies'), ('sacrés','maudits'),('vénérables','abjects'),('fidèles','infidèles'),('croyants','incroyants'),('religieux','irreligieux'),('dévoués','athées')],
           'economic': [('donner','approprier'),('prodigues','cupides'),('générosités','cupidités'),('générosités','avidités'),('généreux','avides'),('généreux','avares'),('généreux', 'pingres')],
           'socio-political':[('honnêtes','canailles'),('honneurs','hontes'),('prodigue','cupide'),('amicaux','hostiles'),('loyaux','fourbes'),('socialistes','capitalistes'),('hôte','parasites'),('amis','ennemi'),('alliés','antagonistes'),('conservateurs','progressistes'),('autoritaires','révolutionnaires')],
           'racist':[('amusants','parasites'),('normal','étrange'),('supériorité','infériorité'),('égalité','inégalité'),('estimables','laids'),('affables','méchants'),('estimables','infâmes'),('sympathies','haines'),('accepté','refusés'),('meilleurs','pires'),('nationaux','étrangers'),('purs','impurs'),('supérieurs','inférieures'),('purs','infects'),('propres','sales')],
           'conspiratorial':[('loyaux','espions'),('honnêtetés','trahisons'),('loyaux','traîtres'),('clair','mystérieux'),('évidents','occultes'),('innocents','périlleux'),('vrai','faux'),('sincères','trompeurs'),('loyaux','déloyaux'),('honnêtes','douteux'),('innocents','criminels'),('publics','secrets'),('ouverts','cachés'),('ostensibles','souterrains'),('amicaux','menaçants'),('clairs','obscurs')],
           'ethic':[('chasteté','érotisme'),('modests','intrigants'),('décente','indécente'),('vertueux','lascifs'),('fidèles','infidèles'),('moraux','immoraux'),('honnêtes','malhonnêtes'),('vertueuse','corrompus'),('chastes','dépravés'),('chastes','charnels'),('purs','dépravée')]
           }

streams_fs = {'religious' : [('spirituel','séculier'),('ange','diable'),('sacrée','profane'),('pieuse','athée'), ('pieuse','païenne'), ('pieuse','idolâtre'), ('pieuse','impie'), ('sacrée','maudit'),('vénérable','abject'),('fidèle','infidèle'),('croyante','incroyante'),('religieuse','irreligieuse'),('dévouée','athée')],
           'economic': [('donner','approprier'),('prodigue','cupide'),('générosité','cupidité'),('générosités','avidité'),('généreuse','avide'),('généreuse','avare'),('généreuse', 'pingre')],
           'socio-political':[('honnête','canaille'),('honneur','honte'),('amicale','hostile'),('loyale','fourbe'),('socialiste','capitaliste'),('hôtesse','parasite'),('amie','ennemie'),('alliée','antagoniste'),('conservatrice','progressiste'),('autoritaire','révolutionnaire')],
           'racist':[('amusante','parasite'),('normal','étrange'),('supériorité','infériorité'),('égalité','inégalité'),('estimable','laide'),('affable','méchante'),('estimable','infâme'),('sympathie','haine'),('acceptés','refusée'),('meilleure','pire'),('nationals','étrangère'),('pure','impur'),('superieure','inférieure'),('pure','infecte'),('propre','sale')],
           'conspiratorial':[('loyale','espion'),('honnêteté','trahison'),('loyale','traîtresse'),('clair','mystérieuse'),('évidente','occulte'),('innocente','périlleuse'),('vrais','fausse'),('sincère','trompeuse'),('loyale','déloyale'),('honnête','douteuse'),('innocente','criminelle'),('public','secrète'),('ostensible','souterraine'),('amicale','menaçante'),('claire','obscure')],
           'ethic':[('chasteté','érotisme'),('modeste','intrigante'),('décente','indécente'),('vertueuse','lascive'),('fidèle','infidèle'),('morale','immorale'),('honnête','malhonnête'),('vertueux','corrompue'),('chaste','dépravée'),('chaste','charnelle'),('pure','dépravée')]
           }


streams_fp = {'religious' : [('spirituel','séculier'),('anges','diables'),('sacrées','profanes'),('pieuses','athées'), ('pieuses','païennes'), ('pieuses','idolâtres'), ('pieuses','impies'), ('sacrées','maudites'),('vénérables','abjects'),('fidèles','infidèles'),('croyantes','incroyantes'),('religieuses','irreligieuses'),('dévouées','athées')],
           'economic': [('donner','approprier'),('prodigues','cupides'), ('générosités','cupidités'),('générosités','avidités'),('généreuses','avides'),('généreuses','avares'),('généreuses', 'pingres')],
           'socio-political':[('honnêtes','canailles'),('honneurs','hontes'),('amicales','hostiles'),('loyales','fourbes'),('socialistes','capitalistes'),('hôtesses','parasites'),('amies','ennemies'),('alliées','antagonistes'),('conservatrices','progressistes'),('autoritaires','révolutionnaires')],
           'racist':[('amusantes','parasites'),('normal','étrange'),('supériorité','infériorité'),('égalité','inégalité'),('estimables','laides'),('affables','méchantes'),('estimables','infâmes'),('sympathies','haines'),('acceptées','refusées'),('meilleures','pires'),('nationales','étrangères'),('pures','impur'),('supérieurses','inférieures'),('pures','infectes'),('propres','sales')],
           'conspiratorial':[('loyales','espions'),('honnêtetés','trahisons'),('loyales','traîtresses'),('clair','mystérieuses'),('évidentes','occultes'),('innocentes','périlleuses'),('vraies','fausses'),('sincères','trompeuses'),('loyales','déloyales'),('honnêtes','douteuses'),('innocentes','criminelles'),('publiques','secrètes'),('ostensibles','souterraines'),('amicales','menaçantes'),('claires','obscures')],
           'ethic':[('chasteté','érotisme'),('modestes','intrigantes'),('décentes','indécentes'),('vertueuses','lascives'),('fidèles','infidèles'),('morales','immorales'),('honnêtes','malhonnêtes'),('vertueuses','corrompues'),('chastes','dépravées'),('chastes','charnelles'),('pures','dépravées')]
           }


words = [('juif', 'm', 's'), ('juifs', 'm', 'p'), ('juive', 'f', 's'), ('juives', 'f', 'p'),
         ('protestant', 'm', 's'), ('protestants', 'm', 'p'), ('protestante', 'f', 's'), ('protestantes', 'f', 'p'),
         ('youpin', 'm', 's'), ('youpins', 'm', 'p'), ('youtre', 'm', 's'), ('youtres', 'm', 'p'),
         ('musulman', 'm', 's'), ('musulmans', 'm', 'p'), ('musulmanne', 'f', 's'), ('musulmannes', 'f', 'p'),
         ('catholique', 'm', 's'), ('catholiques', 'm', 'p'),
         ('chrétien', 'm', 's'), ('chrétiens', 'm', 'p'), ('chrétienne', 'f', 's'), ('chrétiennes', 'f', 'p'),
         ('français', 'm', 's'), ('française', 'f', 's'), ('françaises', 'f', 'p'),
         ('israélite', 'm', 's'), ('israélites', 'm', 's'),
         ('italien', 'm', 's'), ('italiens', 'm', 'p'), ('italienne', 'f', 's'), ('italiennes', 'f', 'p')]


#### ---- FILES PATH ---- ####
### probably you need to change them

# where the embeddings are located
base_all = 'emb/'

# where the results are saved
base = 'bias/'

#### ---- EMBEDDING PARAMETERS ---- ####

### probably you need to change them
### used to construct the file name of the embeddings and to load them

win = 5 # window size
size = 300 # vector dimensions
min_count = 25 # minimal number of occurrences
name = 'WV' # word2vec skip-gram

# ----------------


def writeMat(base, name, Mat):
#    Mat = Mat.todense()
    with open(base + name + '.tsv', 'w') as f:
        w = csv.writer(f, delimiter='\t')
        for row in Mat:
            w.writerow(row)


# COMPUTE THE BIAS DIRECTION

def get_mean_vec(pos,neg,embed):
    pos_array = [embed.wv.word_vec(p, use_norm=True).tolist() for p in pos if p in embed]
    neg_array = [embed.wv.word_vec(n, use_norm=True) * -1 for n in neg if n in embed]
    neg_array = [n.tolist() for n in neg_array]
    arr = pos_array + neg_array
    mean_array = matutils.unitvec(array(arr).mean(axis=0)).astype(REAL)
    return mean_array

def normalize_val(val, min, max):
    # Normalize to[0, 1]
     range = max - min
     new_val = (val - min) / range

    # Then scale to [min, max]:
     range2 = max - min
     normalized = (new_val*range2) + min
     return normalized

# main function

def mean_bias(word, gender, number):
    S = {}
    bins = [1789, 1830, 1841, 1848, 1855, 1861, 1866, 1870, 1874, 1877, 1880, 1883, 1886, 1889, 1891, 1893, 1895, 1897,
            1899, 1901, 1903, 1905, 1907, 1909, 1911, 1913, 1915]

    print(word)
    S[word] = {}
    if gender == 'm' and number == 's':
        streams = streams_ms
    if gender == 'm' and number == 'p':
        streams = streams_mp
    if gender == 'f' and number == 's':
        streams = streams_fs
    if gender == 'f' and number == 'p':
        streams = streams_fp
    for x in range(len(bins)-1):
        start = bins[x]
        end = bins[x+1]
        S[word][str(start)] = {}
        path = base_all + '3EMB-' + name + '-' + 'win_' + str(win) + '-size_' + str(
                        size) + '-min_count_' + str(min_count) + '-iter_' + str(start) + '-' + str(end)
        embed = KeyedVectors.load(path)
        if word not in embed:
            continue
        embed.init_sims()
        word_vec = embed.wv.word_vec(word, use_norm=True)
        for stream in streams.keys():
            S[word][str(start)][stream] = {}
        for stream in streams.keys():
            pos = [k[0] for k in streams[stream] if k[0] in embed and k[1] in embed]
            neg = [k[1] for k in streams[stream] if k[0] in embed and k[1] in embed]
            print(pos)
            print(neg)
            Cos = []
            Norm_cos = []
            Cos_neg = []
            Norm_cos_neg = []
            for x in range(len(pos)):
                pos_ = pos[x]
                neg_ = neg[x]
                stream_direction = get_mean_vec([pos_],[neg_], embed)
                direction = embed.similar_by_vector(stream_direction,1000000)
                cos = word_vec.dot(stream_direction)
                Cos.append(cos)
                max_ = direction[0][1]
                min_ = direction[-1][1]
                norm_cos = normalize_val(cos, min_, max_)
                Norm_cos.append(norm_cos)
                print(word,stream,pos_,neg_,cos,norm_cos)

                stream_direction = get_mean_vec([neg_], [pos_], embed)
                cos = word_vec.dot(stream_direction)
                Cos_neg.append(cos)
                norm_cos = normalize_val(cos, min_, max_)
                Norm_cos_neg.append(norm_cos)

            S[word][str(start)][stream]['pos'] = np.mean(Cos)
            S[word][str(start)][stream]['norm_pos'] = np.mean(Norm_cos)
            S[word][str(start)][stream]['neg'] = np.mean(Cos_neg)
            S[word][str(start)][stream]['norm_neg'] = np.mean(Norm_cos_neg)

        with open('bias/All_mean_bias2.pkl','wb') as f:
            pickle.dump(S,f)


# EXAMPLE
mean_bias('juif','m','s')
