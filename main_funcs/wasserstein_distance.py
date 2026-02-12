from main_funcs.search_openiti import openitiCorpus, openitiTextFull
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import numpy as np
import pandas as pd

class wassersteinPipeline():
    """Pipeline that takes an openITI corpus path and metadata and given a set of keywords
    calculates the wasserstein distance between every pair of texts in the corpus.
    Class consists of series of functions that correspond to sequential steps in the pipeline"""
    
    def __init__(self, meta_path, corpus_base_path, min_date=0, max_date=1500, book_list=None):
        """Initialise with a list of paths ready to apply the stages of the pipeline
        MAY NEED TO KEEP THE METADATA AROUND FOR LATER OUTPUTS"""
        self.corpus_path_dict = openitiCorpus(meta_path, corpus_base_path, pri_only=True, min_date=min_date, max_date=max_date).path_dict

        if book_list:
            self.filter_by_book_list(book_list)

    def filter_by_book_list(self, book_list):
        filtered_path_dict = {}
        for book in book_list:
            if book in self.corpus_path_dict.keys():
                filtered_path_dict[book] = self.corpus_path_dict[book]
        
        self.corpus_path_dict = filtered_path_dict

    def prepare_token_offsets(self, regex):
        """Using a regex fetch the token offsets for a 
        For binning and normalisation need to return text length. Text length is needed for normalisation
        prodecure
        Returns: list of dictionaries
        [{"uri": "0000Author.Book", "token_offsets": [000,000,000], "token_length": 000- }]"""
        offset_data = []

        for book_uri, path in tqdm(self.corpus_path_dict.items()):

            openiti_text = openitiTextFull(path)
            token_offsets = openiti_text.finditer_tokens(regex)
            # Only add the data if it returns something
            if len(token_offsets) > 0:
                data = {"uri": book_uri,
                        "token_offsets": token_offsets ,
                        "token_length": openiti_text.token_text_length()
                        }
                offset_data.append(data)
        
        return offset_data
    
    def normalise_offsets(self, offset_data):
        """Take the token_offsets and normalise them by dividing them by the book length"""
        for data in offset_data:
            normalised_array = [i / data["token_length"] for i in data["token_offsets"]]
            data["normalised_offsets"] = normalised_array
        
        return offset_data
            


    def build_bins(self, offset_data, bins, normalise_density=True):
        """Take the offset_data (a list of dictionaries) and remap normalised array for the token offsets"""

        for data in tqdm(offset_data):
            array = np.array(data["normalised_offsets"])

            # Range 0,1 says always expect values between 0 and 1 - as we normalised that is expected
            counts, bin_edges = np.histogram(array, bins=bins, range=(0,1))
            
            # If normalising density, then divide the count in each bin by the total counts overall
            if normalise_density:
                counts = counts / counts.sum()
            
            data["offset_bins"] = counts
        
        return offset_data


    def calculate_wasserstein_pairwise(self, offset_data):
        """Calculate pairwise wassertein for every pair in the corpus"""
        uris = []
        pairwise_data = []
        for data_1 in tqdm(offset_data):
            uri_1 = data_1["uri"]

            for data_2 in offset_data:                
                uri_2 = data_2["uri"]
                if uri_1 == uri_2:
                    continue
                elif uri_2 in uris:
                    continue
                else:
                    uris.append(uri_1)
                    distance = wasserstein_distance(data_1["offset_bins"], data_2["offset_bins"])
                    data_out = {"book1": uri_1,
                                "book2": uri_2,
                                "wasserstein_distance": distance}
                    pairwise_data.append(data_out)
        
        return pairwise_data
    
    def produce_corpus_pairwise(self, regex, bins, csv_out=None):

        data = self.prepare_token_offsets(regex)
        data = self.normalise_offsets(data)
        data = self.build_bins(data, bins)
        pairwise_data = self.calculate_wasserstein_pairwise(data)

        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df = pairwise_df.sort_values(by=["wasserstein_distance"], ascending=False)

        if csv_out:
            pairwise_df.to_csv(csv_out, index=False)
        
        return pairwise_df

                

        
    


