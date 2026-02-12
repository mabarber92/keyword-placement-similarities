from openiti.helper.funcs import read_text, text_cleaner
from openiti.helper.ara import normalize_ara_heavy, tokenize
import re
import os
import pandas as pd

class openitiTextFull():
    """A class for handling an OpenITI text as a full text - primarily for performing searches"""
    def __init__ (self, file_path, clean=False):
        """Read the text into the object using a file path"""
        
        # Initiate the ms_pattern to be used across the class
        self.ms_pattern = r"ms\d+"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")

        # Read in OpenITI text - split off header        
        self.mARkdown_text = read_text(file_path, remove_header=True)

        if clean:
            self.mARkdown_text = text_cleaner(self.mARkdown_text)
    
    def create_token_mapping(self, token_regex=r"\W\w+"):
        """For whole text create a mapping for the character start point of each token in the text"""
        
        token_mapping = {}
        all_tokens = re.finditer(token_regex, self.mARkdown_text)
        for idx, token in enumerate(all_tokens):
            token_mapping[token.start(0)] = idx
        
        return token_mapping
    
    def finditer_tokens(self, regex):
        """Use regex and finditer to fetch token positions of the regex, rather than than
        index positions. Return a list of token start positions"""

        token_mapping = self.create_token_mapping()
        matches = re.finditer(regex, self.mARkdown_text)

        # Loop through results and get the token offsets
        token_offsets = []
        for match in matches:
            
            # Get character start
            start = match.start(0)

            # Get the number of tokens closest to the token mapping in the dict
            while start not in token_mapping.keys():
                start -= 1
            
            token_offset = token_mapping[start]

            token_offsets.append(token_offset)
        
        return token_offsets
    
    def token_text_length(self):
        """Get the full token length of the text"""
        return len(self.mARkdown_text.split())
    
    def return_cleaned_text(self, normalise=False):
        """Return a text cleaned using openiti func"""
        text = text_cleaner(self.mARkdown_text)
        if normalise:
            text = normalize_ara_heavy(text)
        return text
    
    def return_cleaned_tokenized(self, normalise=False):
        """Use OpenITI tokenizer on a normalised or non-normalised text"""
        text = self.return_cleaned_text(normalise=normalise)
        tokens, token_starts, token_ends = tokenize(text)
        return tokens

    
            

class openitiCorpus():
    """Take corpus base path and a metadata tsv, create paths. Perform actions
    on those texts as openITI objects"""
    def __init__ (self, meta_tsv, base_path, pri_only = True, min_date = 0, max_date = 1500):
        """Initiate with a dictionary of URI-path pairs"""

        meta_df = self.load_and_filter(meta_tsv, pri_only, min_date, max_date)

        self.path_dict = self.build_path_dict(meta_df, base_path)
    
    def load_and_filter(self, meta_tsv, pri_only, min_date, max_date):
        print(max_date)
        meta_df = pd.read_csv(meta_tsv, sep="\t")
        
        if pri_only:
            meta_df = meta_df[meta_df["status"]=="pri"]
        
        
        meta_df = meta_df[meta_df["date"].ge(min_date)]
        meta_df = meta_df[meta_df["date"].le(max_date)]

        return meta_df
        

    def build_path_dict(self, meta_df, base_path):
        """Take a path to a meta tsv and a OpenITI base path
        Build a dictionary
        Returns: path_dict
        {"bookURI": "path"}
        """




        meta_dict = meta_df[["book", "local_path"]].to_dict("records")
        path_dict = {}

        for meta in meta_dict:
            full_path = os.path.join(base_path, meta["local_path"].split("../")[-1])
            path_dict[meta["book"]] = full_path
        
        return path_dict
    
    def return_path_list(self):
        """Take the values of the path dict and return them as a list of paths"""
        return self.path_dict.values()
    
    def fetch_path_for_books(self, book_uris):
        """a list of uris returns a list of paths"""
        if type(book_uris) == str:
            return self.path_dict[book_uris]
        elif type(book_uris) == list:
            file_paths = []
            for book in book_uris:
                file_paths.append(self.path_dict[book])
            return file_paths



if __name__ == "__main__":

    meta_path = "D:/Corpus Stats/2023/OpenITI_metadata_2023-1-8.csv"
    base_path = "D:/OpenITI Corpus/corpus_2023_1_8"
    test_regex = r"غلاء|محن"
    book_uri = "0845Maqrizi.IghathaUmma"

    text_path = openitiCorpus(meta_path, base_path).fetch_path_for_book(book_uri)
    openiti_text = openitiTextFull(text_path)
    print(openiti_text.finditer_tokens(test_regex))