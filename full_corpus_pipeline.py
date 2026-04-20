from main_funcs.tfidf_funcs import corpusIDF, tfidfOpenITI
from main_funcs.compare_tfidf import tfidfSimilarity
import yaml
import os
import shutil

CONFIG = "./full_corpus_config.yml"

class tfidfSimilarityPipeline():
    def __init__(self, use_config=True):

        if use_config:
            self.load_config()
        self.idf_path = None
        self.tfidf_dir = None

    def load_config(self, config=CONFIG):
        """Load the full corpus config file"""
        print(f"Loading config from {config}")

        with open(config) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        
        self.pipeline_components = data["pipeline_stages"]
        print(self.pipeline_components)
        self.min_date = data["min_date"]
        self.max_date = data["max_date"]

        self.openiti_path = data["openiti_path"]
        self.meta_path = data["meta_path"]

        self.out_path = data["out_path"]

        if "BPE_tokenizer" in data.keys():
            self.BPE_tokenizer = data["BPE_tokenizer"]
        else:
            self.BPE_tokenizer = None
        
        # Create a copy of the config in the destination - to keep a record of settings used
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        stored_config = os.path.join(self.out_path, "run_config.yml")
        shutil.copyfile(config, stored_config)
    
    def _check_create_path(self, path, mkdir=True):
        if not os.path.exists(path):
            print(path)
            if mkdir:
                os.mkdir(path)
            return False
        else:
            return True

    def _qualify_run_status(self, path, overwrite, mkdir=True):
        
        run_process = False
        if self._check_create_path(path, mkdir=mkdir):
            if overwrite:
                run_process=True
        else:
            run_process = True
        return run_process

    def run_pipeline(self):
        # Note - not terribly flexible - assumes pipeline stages must be there to update paths with overwrite false - a little brittle
        # If idf has not been run or overwrite set to true run idf
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)

        if "full_corpus_idf" in self.pipeline_components.keys():
            settings = self.pipeline_components["full_corpus_idf"]
            settings_list = list(settings.keys())
            
            # Set defaults if values are not present
            if "overwrite" not in settings_list:
                settings["overwrite"] = False
            if "min_df" not in settings_list:
                settings["min_df"] = 0
            if "max_df" not in settings_list:
                settings["max_df"] = 0.8
            if "language" not in settings_list:
                settings["language"] = "ara"
            
            out_path = os.path.join(self.out_path, "idf.json")
            self.idf_path = out_path
            
            if self._qualify_run_status(out_path, settings["overwrite"], mkdir=False):
                corpus_idf = corpusIDF(self.meta_path, self.openiti_path, settings["language"], min_date = self.min_date, max_date=self.max_date, BPE_tokenizer=self.BPE_tokenizer)
                corpus_idf.create_and_store_batched(out_path)
            
        if "tfidf" in self.pipeline_components.keys():
            
            
            
            settings = self.pipeline_components["tfidf"]
            settings_list = list(settings.keys())
            # Set defaults if not present
            if "overwrite" not in settings:
                settings["overwrite"] = False
            if "normalise" not in settings:
                settings["normalise"] = "log"
            
            out_path = os.path.join(self.out_path, "tfidf_csvs/")
            self.tfidf_dir = out_path

            if self._qualify_run_status(out_path, settings["overwrite"]):
                tfidf_obj = tfidfOpenITI(self.meta_path, self.openiti_path, self.idf_path, multiprocess=True,  BPE_tokenizer=self.BPE_tokenizer)
                tfidf_obj.csv_pipeline(out_path, separate_uris=True, date_filter=[self.min_date, self.max_date], normalise=settings["normalise"])
        
        if "cosine_similarity" in self.pipeline_components.keys():

            settings = self.pipeline_components["cosine_similarity"]
            settings_list = list(settings.keys())

            if "overwrite" not in settings:
                settings["overwrite"] = False
            if "book_focus" not in settings:
                settings["book_focus"] = None

            out_path = os.path.join(self.out_path, "pairwise_similarities")

            if self._qualify_run_status(out_path, settings["overwrite"]):
                similarity_calculator = tfidfSimilarity(self.tfidf_dir)

                if settings["book_focus"] is not None or settings["book_focus"] != []:
                    for book in settings["book_focus"]:
                        csv_path = os.path.join(out_path, f"{book}-similarities.csv")
                        similarity_calculator.one_to_all_csv(book, csv_path)
                else:
                    similarity_calculator.compare_all_pairwise(out_path)
        
        print("Done!")

if __name__ == "__main__":
    tfidfSimilarityPipeline().run_pipeline()
            



