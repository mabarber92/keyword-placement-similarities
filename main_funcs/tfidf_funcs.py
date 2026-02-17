from main_funcs.search_openiti import openitiCorpus, openitiTextFull
from tqdm import tqdm
import math
import json
import os
import pandas as pd
from collections import Counter
from multiprocessing import Pool

class tfidfOpenITI():
    """Take a path dict and texts in an in group and produce a tfidf token list
    where the tf is the frequency in the in_texts (one or many - if multiple texts we concatenate them) and idf is the number
    of books in the corpus that mention the term - using a idf json"""
    def __init__(self, meta_tsv, corpus_base_path, idf_json_path):
        
        # Load the idf data
        self.load_idf_data(idf_json_path)

        # Get the paths to the in_texts
        self.openiti_corpus = openitiCorpus(meta_tsv, corpus_base_path)
    
    def load_idf_data(self, idf_json_path):

        with open(idf_json_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        self.n_docs = data["n_docs"]
        self.idf_dict = data["idfs"]
    
    def compute_frequencies(self, token_list):
        """Take a list of tokens and compute the frequencies in the token set
        Returns a list of dictionaries - able to be converted into a df"""
        
        print("Counting frequencies")
        counts = Counter(token_list)

        freq_list = []
        for token in counts.keys():
            row = {"token": token, "frequency": counts[token]}
            freq_list.append(row)
        
        return freq_list

    def calculate_tfidf(self, freq_list):
        """Using a list of dictionaries of tokens and their frequencies perform a 
        look up in the idf data and use that to compute tf-idf. Add as 
        a new item to the dictionary in the row"""

        print("Calculating tfidf")
        computed_data = []
        for row in tqdm(freq_list):
            token = row["token"]
            idf_score = self.idf_dict.get(token, None)

            # if idf score is returned then we add that token to the list 
            # (otherwise we discard it - these are tokens that tripped min or max_df when idf was computed)
            # ADD - AS test - skip anything where the score is 1 or less - this filter better applied by using max_df
            if idf_score is not None:
                tfidf_score = row["frequency"] * idf_score
                row["tfidf"] = tfidf_score
                computed_data.append(row)

        return computed_data 

    def sort_and_filter(self, tfidf_data, top_terms=None):
        """Sort by tfidf score and if top_terms is not None, take the top n terms"""
        
        tfidf_data = sorted(tfidf_data, key= lambda d: d['tfidf'], reverse=True)

        if top_terms is not None:
            return tfidf_data[:top_terms]
        else:
            return tfidf_data

    def compute_tfidfs(self, text_paths, top_terms=None):
        """Compute the document frequencies for supplied text paths"""
        
        token_list = []
        for path in text_paths:
            openiti_obj = openitiTextFull(path)
            tokens = openiti_obj.return_cleaned_tokenized(normalise=True)
            token_list.extend(tokens)
        
        # Calculate frequency and use that to calculate tf-idf
        freq_list = self.compute_frequencies(token_list)
        tfidf_data = self.calculate_tfidf(freq_list)

        # Sort by tfidf score and filter on top n terms
        tfidf_data = self.sort_and_filter(tfidf_data, top_terms=top_terms)

        return tfidf_data


    def fit_transform(self, uri_list, separate_uris=False, top_terms=None):
        """Fit transform tf-idf model to a list of uris - treat all as in docs
        If separate_uris, produce a separate output for each uri
        Returns: a dict of lists of dictionaries (ready to be converted into dfs)
        {"000Author.Book": [TFIDF FREQ DATA]...}"""
        
        self.data_out = {}
        
        # Fetch the relevant text paths
        if not separate_uris:
            text_paths = self.openiti_corpus.fetch_path_for_books(uri_list)
            tfidf_data = self.compute_tfidfs(text_paths, top_terms=top_terms)
            self.data_out[f"{len(uri_list)}-books-tfidf"] = tfidf_data
        
        else:
            for uri in uri_list:
                text_path = self.openiti_corpus.path_dict[uri]
                tfidf_data = self.compute_tfidfs([text_path], top_terms=top_terms)
                self.data_out[uri] = tfidf_data
        
        return self.data_out

    def write_csvs(self, dir):
        """Take self.data_out and write the data there to csvs in a specified directory"""
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        for data_name, data in self.data_out.items():
            df = pd.DataFrame(data)
            file_name = f"{data_name}.csv"
            path = os.path.join(dir, file_name)
            df.to_csv(path, index=False, encoding='utf-8-sig')
    
    def csv_pipeline(self, uri_list, dir, separate_uris=False, top_terms=None):
        """Run the full pipeline and write out csvs"""
        
        self.fit_transform(uri_list, separate_uris=separate_uris, top_terms=top_terms)
        self.write_csvs(dir)

    


class corpusIDF():
    """Single use function to build an IDF of the entire OpenITI corpus
    Produces a json that can be passed to the tfidf class for matching
    IDF values
    We need this function because computing sklearn's TF-IDF on the full
    corpus would blow out memory + no need to compute every single time we
    want to get a TF-IDF score for a specific document"""
    def __init__(self, meta_tsv, corpus_base_path, language, min_df=0, max_df=0.8, pri_only = True, min_date=0, max_date=1500, book_list=[]):
        """Get the list of file paths to be processed into an IDF representation"""
        
        # Set the file paths
        self.set_file_paths(meta_tsv, corpus_base_path, language, pri_only, min_date, max_date, book_list)
        self.n_docs = len(self.file_paths)

        # Set the parameters to run
        self.min_df = min_df
        self.max_df = max_df



    
    def set_file_paths(self, meta_tsv, corpus_base_path, language, pri_only, min_date, max_date, book_list):
        """Initiate an openitiCorpus object. Return a list of file paths based
        on specified parameters"""
        corpus_obj = openitiCorpus(meta_tsv, corpus_base_path, language, pri_only, min_date, max_date)

        if len(book_list) > 0:
            self.file_paths = corpus_obj.fetch_path_for_books(book_list)
        else:
            self.file_paths = corpus_obj.return_path_list()

    def load_fetch_tokens(self, text_path, normalise=True):
        """Take a path to an openITI text and load it as a set of unique tokens"""
        openiti_tokens = openitiTextFull(text_path).return_cleaned_tokenized(normalise=normalise)
        unique_tokens = set(openiti_tokens)
        return unique_tokens

    # def add_tokens_to_df(self, unique_tokens):
    #     for token in unique_tokens:
    #         self.dfs[token] = self.dfs.get(token, 0) + 1

    def populate_dfs(self, file_paths, multiprocess=True):
        """Go through paths to the texts, load the texts and populate the df dict"""
        
        dfs = Counter()
        
        if not multiprocess:
            for path in tqdm(file_paths):
                unique_tokens = self.load_fetch_tokens(path)
                dfs.update(unique_tokens)
        
        else:
            for path in file_paths:
                unique_tokens = self.load_fetch_tokens(path)
                dfs.update(unique_tokens)

        return dfs
            

    def apply_filters(self):
        """Using the set filters, filter the frequency dict"""
        
        if type(self.max_df) == float:
            self.max_df = int(self.max_df * self.n_docs)

        for token, count in list(self.dfs.items()):
            if count < self.min_df or count > self.max_df:
                del self.dfs[token]
    
    def compute_idf(self):
        """Take self.dfs and calculate the idf - store as separate dict"""
        self.idfs = {}
        
        for token, count in self.dfs.items():
            idf = math.log((self.n_docs +1) / (count + 1)) + 1
            self.idfs[token] = idf
    
    def write_json(self, json_path):

        data = {
            "n_docs": self.n_docs,
            "idfs": self.idfs
        }

        json_data = json.dumps(data, indent=2, ensure_ascii=False)

        with open(json_path, "w", encoding='utf-8') as f:
            f.write(json_data)

    def create_and_store_idf(self, json_path):
        """Run the full pipeline and export as json"""
        
        self.dfs = dict(self.populate_dfs(self.file_paths, multiprocess=False))
        self.apply_filters()
        self.compute_idf()
        self.write_json(json_path)
    

    def create_batch(self, path_list, batch_size):
        for i in range(0, len(path_list), batch_size):
            yield path_list[i:i+batch_size]

    def create_and_store_batched(self, json_path, batch_size, pool_size):
        """Run create_and_store_idf but batching texts and parrlelising to deal with large corpus"""

        self.dfs = Counter()
        n_batches = math.ceil(len(self.file_paths) / batch_size)


        with Pool(processes=pool_size) as pool:
            for batch in tqdm(pool.imap_unordered(self.populate_dfs, self.create_batch(self.file_paths, batch_size)),
                              total=n_batches, desc="Batches", unit="batch"):
                self.dfs.update(batch)
        
        self.dfs = dict(self.dfs)
        self.apply_filters()
        self.compute_idf()
        self.write_json(json_path)


