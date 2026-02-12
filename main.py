
from main_funcs.wasserstein_distance import wassersteinPipeline
import json
from main_funcs.tfidf_funcs import corpusIDF, tfidfOpenITI

if __name__ == "__main__":

    meta_path = "E:/Corpus Stats/2023/OpenITI_metadata_2023-1-8.csv"
    base_path = "E:/OpenITI Corpus/corpus_2023_1_8"
    test_regex = r"غلاء|محن"
    test_regex = r"فيها"
    csv_out="test_pairwise.csv"
    csv_dir = "data/evaluate_bins_chronicles/"
    book_list = ["0310Tabari.Tarikh", "0310Tabari.JamicBayan", "0421Miskawayh.HikmaKhalida", "0630IbnAthirCizzDin.Kamil",
                 "0421Miskawayh.Tajarib", "0845Maqrizi.IghathaUmma", "0845Maqrizi.Suluk", "0845Maqrizi.Mawaciz", "0845Maqrizi.ItticazHunafa"]
    idf_json = "test_idf.json"
    tfidf_uris = [book_list[0], book_list[3]]

    # wasserstein = wassersteinPipeline(meta_path, base_path, book_list = book_list)
    # # wasserstein.survey_bin_parameters(test_regex, range(100,500,100), csv_dir)
    # pairwise_data, data = wasserstein.produce_corpus_pairwise(test_regex, 500, csv_out="data/500bins_fiha.csv")

    # json_path = "data/chronicles_data_bins500_fiha.json"
    # json_data = json.dumps(data, indent=4)
    # with open(json_path, "w") as f:
    #     f.write(json_data)

    # corpus_idf = corpusIDF(meta_path, base_path, book_list=book_list)
    # corpus_idf.create_and_store_idf("test_idf.json")

    tfidf = tfidfOpenITI(meta_path, base_path, idf_json)
    tfidf.csv_pipeline(tfidf_uris, "data/tfidf_tests", top_terms=30)
    tfidf.csv_pipeline(tfidf_uris, "data/tfidf_tests", separate_uris=True, top_terms=30)


