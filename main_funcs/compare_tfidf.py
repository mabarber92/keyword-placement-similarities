import os
import re
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np

class tfidfSimilarity():
    def __init__(self, csv_dir):
        """Take a directory of csvs containing tfidf weights and use them to
        compare tfidf between documents"""

        # Init the dictionary for performing the queries and comparisons
        self._uri_csv_dict(csv_dir)

        

    
    def _uri_csv_dict(self, csv_dir):
        """Use a logic to split the csvs into a dict {uri: csv_path}"""
        csv_names = os.listdir(csv_dir)
        if not re.match(r"\d{4}", csv_names[0][:4]):
            # Sanity check - does the filename start with 4 digits - then likely a valid URI
            print(f"Invalid URI found in filename {csv_names[0]}")
            exit()
        self.path_dict = {}
        self.uri_list = []
        for csv_name in csv_names:
            full_path = os.path.join(csv_dir, csv_name)
            uri = csv_name.split(".csv")[0]
            
            if not re.match(r"\d{4}", uri[:4]):
                # Sanity check - does the filename start with 4 digits - then likely a valid URI - otherwise don't add to path dict
                
                print(f"Invalid URI found in filename {csv_name}")
                continue
            self.path_dict[uri] = full_path
            self.uri_list.append(uri)
        print(f"Initialised tfidf df with {len(self.uri_list)} valid uris")
    
    def _load_csv_as_dict(self, text_uri):
        """Load a csv from the path_dict based on specified uri
        Transform to key value pairs where token is key and value is tfidf score"""
        if text_uri not in self.path_dict.keys():
            print(f"{text_uri} not found in supplied directory")
            exit()
        dict_list = pd.read_csv(self.path_dict[text_uri])[["token", "tfidf"]].to_dict("records")
        data = {}
        for row in dict_list:
            data[row["token"]] = row["tfidf"]
        return data

    def _identify_top_tokens(self, weight_pairs, token_list, top_n=10):
        """Take aligned vectors and corresponding token list and get a list of top shared tokens"""
        
        # Convert to numpy vectors
        vec1, vec2 = np.array(weight_pairs[0]), np.array(weight_pairs[1])
        # Normalise the vectors
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

        # Get a contribution for each token as a product of the two normalised scores
        contributions = (vec1/norm1) * (vec2/norm2)

        # Align vectors to tokens
        token_contributions = {token: contributions[i] for i, token in enumerate(token_list)}
        # Sort and take top_n
        top_n = list(sorted(token_contributions.items(), key=lambda x: x[1], reverse=True))[:top_n]
        
        top_n = [tok[0] for tok in top_n]
        
        
        return top_n

    def _align_weights(self, t1, t2):
        """Take two text_uris, load the data, align the weights (filling absent tokens with 0)
        returns: pair of aligned vectors"""
        # Load data for each side of relationship
        data_1 = self._load_csv_as_dict(t1)
        data_2 = self._load_csv_as_dict(t2)

        # Get a list of all tokens
        all_tokens = list(data_1.keys()) + list(data_2.keys())
        all_tokens = list(set(all_tokens))

        # Loop through tokens and produce aligned vectors- filling in zeros
        aligned_weights = [[],[]]
        
        for token in all_tokens:
            weight_1 = data_1.get(token, 0)
            weight_2 = data_2.get(token, 0)
            aligned_weights[0].append(weight_1)
            aligned_weights[1].append(weight_2)

        return aligned_weights, all_tokens

    def fetch_top_shared_toks(self, t1, t2, top_n=20):
        """For a pair of URIs in the directory return the shared tokens"""
        aligned_weights, all_tokens = self._align_weights(t1, t2)
        top_tokens = self._identify_top_tokens(aligned_weights, all_tokens, top_n)
        return top_tokens

    def compare_weights(self, t1, t2):
        """Take two uris of texts and compare the tfidf weights for the tokens"""

        # Create aligned vectors
        aligned_weights, all_tokens = self._align_weights(t1, t2)
        
        if self.top_n_tokens is not None:
            top_tokens = self._identify_top_tokens(aligned_weights, all_tokens, self.top_n_tokens)
        else:
            top_tokens = None
        
        # Compute cosine similarity
        similarity = 1- cosine(aligned_weights[0], aligned_weights[1])

        return similarity, top_tokens
    
    def compare_one_to_all(self, main_uri, top_tok_joiner = "_"):
        """Compare a main_uri with all of the other uris in the supplied directory"""

        print(f"Running similarities for {main_uri}")
        # Remove main URI from the list to be compared against (don't compare it with itself)
        comparison_list = self.uri_list.copy()
        comparison_list.remove(main_uri)

        # Compute similarities
        data_out = []

        # TODO - Add ability to parallelise this
        for uri in tqdm(comparison_list):
            similarity, top_tokens = self.compare_weights(main_uri, uri)
            data_dict = {"b1": main_uri, 
                        "b2": uri,
                        "similarity": similarity}
            
            if top_tokens is not None:
                # Join the token list using the specified joiner
                top_tokens = top_tok_joiner.join(top_tokens)
                data_dict[f"top_{self.top_n_tokens}_tokens"] = top_tokens
            
            data_out.append(data_dict)
        
        # Transform to df and sort
        data_out = pd.DataFrame(data_out)
        
        data_out = data_out.sort_values(by=["similarity"], ascending = False)

        return data_out



    def one_to_all_csv(self, main_uri, csv_out, top_n_tokens=None):
        """Take one uri compare that uri pairwise to every other uri in the given directory return a pairwise csv"""
        self.top_n_tokens = top_n_tokens
        df = self.compare_one_to_all(main_uri)
        df.to_csv(csv_out, index=False)

    def compare_all_pairwise(self, csv_dir_out, top_n_tokens=None):
        """Produce pairwise csvs for every csv in the input directory - return as set of csvs in a directory"""
        # Check dir exists - if not make it
        if not os.path.exists(csv_dir_out):
            os.mkdir(csv_dir_out)
        
        # Set top_no_tokens - for adding token evidence to similarities
        self.top_n_tokens = top_n_tokens

        # Loop through all books and calculate and write pairwise csvs
        for uri in self.uri_list:
            csv_path = os.path.join(csv_dir_out, f"{uri}-tfidf-pairwise.csv")
            self.one_to_all_csv(uri, csv_path)

if __name__ == "__main__":
    tfidf_dir = "../data/full_corpus_runs/800_850_BPE_toks_noprefix/tfidf_csvs"
    
    tfidf_dfs = tfidfSimilarity(tfidf_dir)
    print(tfidf_dfs.fetch_top_shared_toks("0845Maqrizi.IghathaUmma", "0814IbnZayyat.KawakibSayyara"))
    
    tfidf_dfs.one_to_all_csv("0845Maqrizi.IghathaUmma", "../data/cosine_tests/Ighatha_cosine.csv", top_n_tokens=15)

    # tfidf_dfs.compare_all_pairwise("../data/cosine_tests/", top_n_tokens=10)