"""Small runnable example to train the fallback recommender on a small sample
and print top recommendations for a sample user.
"""
from utils.recommender import WideDeepRecommender
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    train_beh = '/workspace/data/MIND_small/MINDsmall_train/behaviors.tsv'
    val_beh = '/workspace/data/MIND_small/MINDsmall_val/behaviors.tsv'
    news = '/workspace/data/MIND_small/MINDsmall_train/news.tsv'
    r = WideDeepRecommender(train_beh, val_beh, news, device = device)
    r._is_fallback = True
    r.prepare_interactions(nrows_train=5000, nrows_val=1000)
    r.train(epochs=1, batch_size=2048, lr=1e-3, minibatch=True)
    recs = r.recommend_for_user(r.train_inter['user_id'].iloc[0], top_k=5)
    print('Top recommendations:')
    for rec in recs:
        print(rec)
