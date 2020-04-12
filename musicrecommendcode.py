{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'\n",
    "songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas \n",
    "song_df_1 = pandas.read_table(triplets_file,header=None)\n",
    "song_df_1.columns = ['user_id', 'song_id', 'listen_count']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "song_df_2 =  pandas.read_csv(songs_metadata_file)\n",
    "\n",
    "song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on=\"song_id\", how=\"left\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "users = song_df['user_id'].unique()\n",
    "len(users) ## return 365 unique users\n",
    "songs = song_df['song'].unique()\n",
    "len(songs) ## return 5151 unique songs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pm = Recommenders.popularity_recommender_py()\n",
    "pm.create(train_data, 'user_id', 'song')\n",
    "#user the popularity model to make some prediction\n",
    "user_id = users[5]\n",
    "pm.recommend(user_id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Class for Popularity based Recommender System modelclass popularity_recommender_py():    \n",
    "    def __init__(self):        \n",
    "    self.train_data = None        \n",
    "    self.user_id = None        \n",
    "    self.item_id = None        \n",
    "    self.popularity_recommendations = None            \n",
    "    #Create the popularity based recommender system model    \n",
    "    def create(self, train_data, user_id, item_id): \n",
    "        self.train_data = train_data\n",
    "        self.user_id = user_id        \n",
    "        self.item_id = item_id         \n",
    "        \n",
    "        #Get a count of user_ids for each unique song as   recommendation score\n",
    "        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()        \n",
    "        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)            \n",
    "        #Sort the songs based upon recommendation score\n",
    "        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])            \n",
    "        #Generate a recommendation rank based upon score\n",
    "        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')\n",
    "        #Get the top 10 recommendations\n",
    "        self.popularity_recommendations = train_data_sort.head(10)     \n",
    "        #Use the popularity based recommender system model to    \n",
    "        #make recommendations    \n",
    "    def recommend(self, user_id):            \n",
    "        user_recommendations = self.popularity_recommendations                 \n",
    "        #Add user_id column for which the recommendations are being generated        \n",
    "        user_recommendations['user_id'] = user_id            \n",
    "        #Bring user_id column to the front        \n",
    "        cols = user_recommendations.columns.tolist()        \n",
    "        cols = cols[-1:] + cols[:-1]        \n",
    "        user_recommendations = user_recommendations[cols]\n",
    "        return user_recommendations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "is_model = Recommenders.item_similarity_recommender_py()\n",
    "is_model.create(train_data, 'user_id', 'song')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Print the songs for the user in training data\n",
    "user_id = users[5]\n",
    "user_items = is_model.get_user_items(user_id)\n",
    "#\n",
    "print(\"------------------------------------------------------------------------------------\")\n",
    "print(\"Training data songs for the user userid: %s:\" % user_id)\n",
    "print(\"------------------------------------------------------------------------------------\")\n",
    "\n",
    "for user_item in user_items:\n",
    "    print(user_item)\n",
    "\n",
    "print(\"----------------------------------------------------------------------\")\n",
    "print(\"Recommendation process going on:\")\n",
    "print(\"----------------------------------------------------------------------\")\n",
    "\n",
    "#Recommend songs for the user using personalized model\n",
    "is_model.recommend(user_id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#constants defining the dimensions of our User Rating Matrix (URM) MAX_PID = 4 \n",
    "MAX_UID = 5  \n",
    "#Compute SVD of the user ratings matrix \n",
    "def computeSVD(urm, K):     \n",
    "    U, s, Vt = sparsesvd(urm, K)      \n",
    "    dim = (len(s), len(s))     \n",
    "    S = np.zeros(dim, dtype=np.float32)     \n",
    "    for i in range(0, len(s)):         \n",
    "        S[i,i] = mt.sqrt(s[i])      \n",
    "        U = csc_matrix(np.transpose(U), dtype=np.float32)     \n",
    "        S = csc_matrix(S, dtype=np.float32)     \n",
    "        Vt = csc_matrix(Vt, dtype=np.float32)          \n",
    "        return U, S, Vt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Compute estimated rating for the test user\n",
    "def computeEstimatedRatings(urm, U, S, Vt, uTest, K, test):\n",
    "    rightTerm = S*Vt\n",
    "    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)\n",
    "    for userTest in uTest:\n",
    "        prod = U[userTest, :]*rightTerm\n",
    "        #we convert the vector to dense format in order to get the     #indices\n",
    "        #of the movies with the best estimated ratings \n",
    "        estimatedRatings[userTest, :] = prod.todense()\n",
    "        recom = (-estimatedRatings[userTest, :]).argsort()[:250]\n",
    "    return recom\n",
    "\n",
    "#Used in SVD calculation (number of latent factors)\n",
    "K=2\n",
    "#Initialize a sample user rating matrix\n",
    "urm = np.array([[3, 1, 2, 3],[4, 3, 4, 3],[3, 2, 1, 5], [1, 6, 5, 2], [5, 0,0 , 0]])\n",
    "urm = csc_matrix(urm, dtype=np.float32)\n",
    "#Compute SVD of the input user ratings matrix\n",
    "U, S, Vt = computeSVD(urm, K)\n",
    "#Test user set as user_id 4 with ratings [0, 0, 5, 0]\n",
    "uTest = [4]\n",
    "print(\"User id for whom recommendations are needed: %d\" % uTest[0])\n",
    "#Get estimated rating for test user\n",
    "print(\"Predictied ratings:\")\n",
    "uTest_recommended_items = computeEstimatedRatings(urm, U, S, Vt, uTest, K, True)\n",
    "print(uTest_recommended_items)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
