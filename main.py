
from main_funcs.wasserstein_distance import wassersteinPipeline

if __name__ == "__main__":

    meta_path = "D:/Corpus Stats/2023/OpenITI_metadata_2023-1-8.csv"
    base_path = "D:/OpenITI Corpus/corpus_2023_1_8"
    test_regex = r"غلاء|محن"
    test_regex = r"سنة|دخل|فيها"
    csv_out="test_pairwise.csv"
    book_list = ["0310Tabari.Tarikh", "0310Tabari.JamicBayan", "0421Miskawayh.HikmaKhalida",
                 "0421Miskawayh.Tajarib", "0845Maqrizi.IghathaUmma", "0845Maqrizi.Suluk", "0845Maqrizi.Mawaciz", "0845Maqrizi.IttiaczHunafa"]

    wasserstein = wassersteinPipeline(meta_path, base_path, book_list = book_list)
    wasserstein.produce_corpus_pairwise(test_regex, bins=30, csv_out=csv_out)


