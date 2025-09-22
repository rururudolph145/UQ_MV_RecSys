"""Reusable WideDeepRecommender module for MIND data.

This module implements:
- WideDeepRecommender class (same behavior as the notebook version)
- save/load utilities for the fallback PyTorch model
- minibatch training for the fallback model

Note: If `pytorch_widedeep` is available the class will prefer that path for training.
"""
from pathlib import Path
from typing import Optional, List
import json, ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class WideDeepRecommender:
    def __init__(self, train_beh_path: str, val_beh_path: str, news_path: str, device: Optional[str] = None):
        self.train_beh_path = train_beh_path
        self.val_beh_path = val_beh_path
        self.news_path = news_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocessor = None
        self.news_df = None
        self._is_fallback = False
        print(device)

    def _detect_beh_columns(self, path: str):
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.strip():
                    parts = line.strip().split('\t')
                    return len(parts)
        return 0

    def load_behaviors(self, path: str, nrows: Optional[int] = None):
        ncols = self._detect_beh_columns(path)
        if ncols >= 5:
            names = ['impression_id','user_id','impression_time','history','impressions']
        else:
            names = ['impression_time','user_id','history','impressions']
        df = pd.read_csv(path, sep='\t', header=None, names=names, dtype=str, nrows=nrows, quoting=3)
        def parse_history(x):
            if pd.isna(x) or str(x).strip() in ['', '-']:
                return []
            return str(x).split()
        df['history'] = df.get('history', pd.Series(['']*len(df))).apply(parse_history)
        df['raw_impression'] = df.get('impressions', pd.Series(['']*len(df))).fillna('')
        rows = []
        for _, r in df.iterrows():
            raw = str(r['raw_impression']).strip()
            if raw == '':
                continue
            for token in raw.split():
                if '-' in token:
                    news, lbl = token.rsplit('-',1)
                    try:
                        lbl = int(lbl)
                    except Exception:
                        lbl = None
                else:
                    news, lbl = token, None
                rows.append({'user_id': r.get('user_id', None), 'news_id': news, 'label': lbl})
        return pd.DataFrame(rows)

    def load_news(self, path: str, nrows: Optional[int] = None):
        col_names = ['news_id','category','subcategory','title','abstract','url','entities','concepts']
        df = pd.read_csv(path, sep='\t', header=None, names=col_names, nrows=nrows, dtype=str, quoting=3)
        def safe_parse(s):
            if pd.isna(s) or str(s).strip() == '':
                return []
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return []
        df['entities_parsed'] = df['entities'].apply(safe_parse)
        def extract_labels(x):
            if isinstance(x, list):
                return [it.get('Label') for it in x if isinstance(it, dict) and it.get('Label')]
            return []
        df['entity_labels'] = df['entities_parsed'].apply(extract_labels)
        df['title_abstract'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
        self.news_df = df
        return df

    def prepare_interactions(self, nrows_train: Optional[int] = None, nrows_val: Optional[int] = None):
        train_df = self.load_behaviors(self.train_beh_path, nrows=nrows_train)
        val_df = self.load_behaviors(self.val_beh_path, nrows=nrows_val)
        self.load_news(self.news_path)
        # keep only interactions where news exists in news_df
        valid_news = set(self.news_df['news_id'])
        train_df = train_df[train_df['news_id'].isin(valid_news)].copy()
        val_df = val_df[val_df['news_id'].isin(valid_news)].copy()
        train_df['label'] = train_df['label'].astype('Int64').fillna(0).astype(int)
        val_df['label'] = val_df['label'].astype('Int64').fillna(0).astype(int)
        self.train_inter = train_df.reset_index(drop=True)
        self.val_inter = val_df.reset_index(drop=True)
        return self.train_inter, self.val_inter

    # --- training ---
    def train(self, epochs: int = 3, batch_size: int = 1024, lr: float = 1e-3, minibatch: bool = True):
        if getattr(self, 'train_inter', None) is None:
            self.prepare_interactions()
        try:
            # Try widedeep path
            from pytorch_widedeep.preprocessing import TabPreprocessor, WidePreprocessor
            from pytorch_widedeep.models import Wide, TabMlp, WideDeep
            from pytorch_widedeep import Trainer
            X_train = self.train_inter[['user_id','news_id']].astype(str)
            X_val = self.val_inter[['user_id','news_id']].astype(str)
            y_train = self.train_inter['label'].values
            y_val = self.val_inter['label'].values
            tab_preprocessor = TabPreprocessor(embed_cols=['user_id','news_id'], continuous_cols=[])
            X_tab_train = tab_preprocessor.fit_transform(X_train)
            X_tab_val = tab_preprocessor.transform(X_val)
            wide_preprocessor = WidePreprocessor(categorical_cols=['user_id','news_id'])
            X_wide_train = wide_preprocessor.fit_transform(X_train)
            X_wide_val = wide_preprocessor.transform(X_val)
            wide = Wide(wide_dim=X_wide_train.shape[1])
            deeptabular = TabMlp(mlp_hidden_dims=[64,32], dropout=0.2)
            model = WideDeep(wide=wide, deeptabular=deeptabular)
            trainer = Trainer(model, objective='binary', metrics=['auc','accuracy'], use_cuda=(self.device=='cuda'))
            trainer.fit(X_tab=X_tab_train, X_wide=X_wide_train, target=y_train,
                        X_tab_val=X_tab_val, X_wide_val=X_wide_val, val_target=y_val,
                        n_epochs=epochs, batch_size=batch_size, lr=lr)
            self.model = model
            self.preprocessor = {'tab': tab_preprocessor, 'wide': wide_preprocessor}
            print('Trained pytorch_widedeep model.')
            return True
        except Exception as e:
            print('pytorch_widedeep not available or failed:', e)
            return self._train_fallback(epochs=epochs, batch_size=batch_size, lr=lr, minibatch=minibatch)

    def _train_fallback(self, epochs=3, batch_size=1024, lr=1e-3, minibatch=True):
        train = self.train_inter
        val = self.val_inter
        users = pd.concat([train['user_id'], val['user_id']]).unique()
        news = self.news_df['news_id'].unique()
        user2idx = {u:i for i,u in enumerate(users)}
        news2idx = {n:i for i,n in enumerate(news)}
        train['u_idx'] = train['user_id'].map(user2idx)
        train['n_idx'] = train['news_id'].map(news2idx)
        val['u_idx'] = val['user_id'].map(user2idx).fillna(-1).astype(int)
        val['n_idx'] = val['news_id'].map(news2idx).fillna(-1).astype(int)
        n_users = len(user2idx)
        n_news = len(news2idx)
        emb_dim = 64
        class SimpleRecModel(nn.Module):
            def __init__(self, n_users, n_news, emb_dim):
                super().__init__()
                self.u_emb = nn.Embedding(n_users, emb_dim)
                self.n_emb = nn.Embedding(n_news, emb_dim)
                self.out = nn.Linear(emb_dim*2, 1)
            def forward(self, u_idx, n_idx):
                u = self.u_emb(u_idx)
                n = self.n_emb(n_idx)
                x = torch.cat([u,n], dim=1)
                return self.out(x).squeeze(1)
        model = SimpleRecModel(n_users, n_news, emb_dim).to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=lr)
        # prepare arrays
        u_arr = train['u_idx'].values
        n_arr = train['n_idx'].values
        y_arr = train['label'].values.astype(np.float32)
        val_u = val['u_idx'].values
        val_n = val['n_idx'].values
        val_y = val['label'].values.astype(np.float32)
        # minibatch training
        import math
        num_samples = len(u_arr)
        indices = np.arange(num_samples)
        for epoch in range(epochs):
            np.random.shuffle(indices)
            model.train()
            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                end = min(start+batch_size, num_samples)
                batch_idx = indices[start:end]
                u_b = torch.tensor(u_arr[batch_idx], dtype=torch.long, device=self.device)
                n_b = torch.tensor(n_arr[batch_idx], dtype=torch.long, device=self.device)
                y_b = torch.tensor(y_arr[batch_idx], dtype=torch.float32, device=self.device)
                opt.zero_grad()
                logits = model(u_b, n_b)
                loss = loss_fn(logits, y_b)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(batch_idx)
            epoch_loss /= num_samples
            # validation
            model.eval()
            with torch.no_grad():
                v_u = torch.tensor(val_u, dtype=torch.long, device=self.device)
                v_n = torch.tensor(val_n, dtype=torch.long, device=self.device)
                v_logits = model(v_u, v_n)
                v_loss = loss_fn(v_logits, torch.tensor(val_y, dtype=torch.float32, device=self.device)).item()
                preds = (torch.sigmoid(v_logits) > 0.5).float().cpu().numpy()
                acc = (preds == val_y).mean()
            print(f'Epoch {epoch+1}/{epochs} train_loss={epoch_loss:.4f} val_loss={v_loss:.4f} val_acc={acc:.4f}')
        # store
        self.model = model
        self.preprocessor = {'user2idx': user2idx, 'news2idx': news2idx}
        self._is_fallback = True
        return True

    # --- persist/load fallback model ---
    def save_fallback(self, path_prefix: str):
        assert getattr(self, '_is_fallback', False), 'only fallback model saving implemented'
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f'{path_prefix}_model.pt')
        # save preprocessor maps
        import pickle
        with open(f'{path_prefix}_prep.pkl','wb') as fh:
            pickle.dump(self.preprocessor, fh)
        print('Saved fallback model and preprocessor')

    def load_fallback(self, path_prefix: str):
        import pickle
        state = torch.load(f'{path_prefix}_model.pt', map_location=self.device)
        with open(f'{path_prefix}_prep.pkl','rb') as fh:
            prep = pickle.load(fh)
        # recreate model shape
        user2idx = prep['user2idx']
        news2idx = prep['news2idx']
        n_users = len(user2idx)
        n_news = len(news2idx)
        emb_dim = 64
        class SimpleRecModel(nn.Module):
            def __init__(self, n_users, n_news, emb_dim):
                super().__init__()
                self.u_emb = nn.Embedding(n_users, emb_dim)
                self.n_emb = nn.Embedding(n_news, emb_dim)
                self.out = nn.Linear(emb_dim*2, 1)
            def forward(self, u_idx, n_idx):
                u = self.u_emb(u_idx)
                n = self.n_emb(n_idx)
                x = torch.cat([u,n], dim=1)
                return self.out(x).squeeze(1)
        model = SimpleRecModel(n_users, n_news, emb_dim).to(self.device)
        model.load_state_dict(state)
        self.model = model
        self.preprocessor = prep
        self._is_fallback = True
        print('Loaded fallback model')

    # --- recommend ---
    def recommend_for_user(self, user_id: str, top_k: int = 10) -> List[dict]:
        if getattr(self, '_is_fallback', False):
            u2i = self.preprocessor['user2idx']
            n2i = self.preprocessor['news2idx']
            if user_id not in u2i:
                popular = (self.train_inter['news_id'].value_counts().index[:top_k].tolist())
                return [{'news_id': nid, 'title': self.news_df.loc[self.news_df.news_id==nid,'title'].iloc[0]} for nid in popular]
            ui = torch.tensor([u2i[user_id]], dtype=torch.long, device=self.device)
            inv_news = {v:k for k,v in n2i.items()}
            all_n_idx = torch.tensor(list(range(len(n2i))), dtype=torch.long, device=self.device)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(ui.repeat(len(all_n_idx)), all_n_idx)
                scores = torch.sigmoid(logits).cpu().numpy()
            top_idx = scores.argsort()[::-1][:top_k]
            recs = []
            for idx in top_idx:
                nid = inv_news[int(all_n_idx[idx].item())]
                title = self.news_df.loc[self.news_df.news_id==nid, 'title'].iloc[0]
                recs.append({'news_id': nid, 'title': title, 'score': float(scores[idx])})
            return recs
        else:
            try:
                cand = pd.DataFrame({'user_id': [user_id]*len(self.news_df), 'news_id': self.news_df['news_id'].tolist()})
                tab = self.preprocessor['tab'].transform(cand)
                wide = self.preprocessor['wide'].transform(cand)
                from pytorch_widedeep.utils import predict
                preds = predict(self.model, X_tab=tab, X_wide=wide)
                self.news_df['score'] = preds
                top = self.news_df.sort_values('score', ascending=False).head(top_k)
                return top[['news_id','title','score']].to_dict('records')
            except Exception as e:
                print('Failed to use widedeep predict path:', e)
                return []
