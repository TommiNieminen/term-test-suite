import argparse
import itertools
import re
import sentencepiece as spm
from itertools import chain, combinations
import stanza

term_tag_regex = re.compile(r'<term[^>]+tgt="(?P<tgt_term>[^"]+)"[^>]*>(?P<src_term>[^<]+)</term>')

class TermAndTranslations:

    def __init__(self,source_string,target_string,term_index):
        self.source_string = source_string
        self.target_string = target_string
        self.term_index = term_index
        self.source_term = source_string.replace(self.term_index,"")
        self.target_terms = [x for x in target_string.split(self.term_index) if x]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_all_test_pairs(line_terms,source, target):
    term_powerset = powerset(line_terms)
    sentence_pairs = []
    for subset in term_powerset:
        unused_terms = set(line_terms) - set(subset)

        subset_variants = []

        #for unused terms of the subset, only use the first translation variant (usually the original)
        # to keep the amount of test sentence a bit lower
        source_without_unused_terms = source
        target_without_unused_terms = target
        unused_term: TermAndTranslations
        for unused_term in unused_terms:
                source_without_unused_terms = source_without_unused_terms.replace(unused_term.source_string,unused_term.source_term)
                target_without_unused_terms = target_without_unused_terms.replace(unused_term.target_string, unused_term.target_terms[0])
        subset_variants.append((source_without_unused_terms,target_without_unused_terms))

        term : TermAndTranslations
        for term in subset:
            new_variants = []
            for variant in subset_variants:
                for trans in term.target_terms:
                    new_variant_source = variant[0].replace(
                        term.source_string,
                        f'<term id="{term.term_index}" type="src_original_and_tgt_original" ' +
                        f'src="{term.source_term}" tgt="{trans}"> {term.source_term} </term>')
                    new_variant_target = variant[1].replace(
                        term.target_string,
                        f'<term id="{term.term_index}" type="src_original_and_tgt_original " ' +
                        f'src="{term.source_term}" tgt="{trans}"> {trans} </term>')
                    new_variants.append((new_variant_source,new_variant_target))
            subset_variants = new_variants
        sentence_pairs += subset_variants
    return sentence_pairs

def generate_spm_lines(
        all_line_testsets,
        spm_source_model,
        spm_target_model,
        term_start_tag,
        term_end_tag,
        trans_end_tag):
    source_sp_model = spm.SentencePieceProcessor(spm_source_model)
    target_sp_model = spm.SentencePieceProcessor(spm_target_model)

    all_line_sources = []
    for line_testset in all_line_testsets:
        for source,target in line_testset:
            term_tag_regex.findall(source)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a terminology translation test suite from manually annotated files.")
    parser.add_argument("--annotated_file", type=str,
                        help="Tab separated file (source TAB translation) containing term annotations " +
                        "where terms are indicated with pre- and post-fixed ids. Example: source: source" +
                        "with 1term1 or 2another term2 TAB translation with 1term1 or 2another term2" +
                        "Target side can contain alternatives, e.g 1first alt1second alt1")
    parser.add_argument("--spm_output_file", type=str,
                        help="Output source file that has been segmented with the provided source spm model, " +
                        "with terms added using the term tags from the spm model and segmented with the " +
                        "target spm model")
    parser.add_argument("--spm_source_model", type=str,
                        help="spm model for the source language.")
    parser.add_argument("--spm_target_model", type=str,
                        help="spm model for the target language.")
    parser.add_argument("--term_start_tag", type=str, default="<term_start>",
                        help="Tag that is inserted before the source term")
    parser.add_argument("--term_end_tag", type=str, default="<term_end>",
                        help="Tag that is inserted after the source term and before translation lemma")
    parser.add_argument("--trans_end_tag", type=str, default="<trans_end>",
                        help="Tag that is inserted after the translation lemma")
    parser.add_argument("--source_lang", type=str,
                        help="Source language")
    parser.add_argument("--target_lang", type=str,
                        help="Target language, used with Stanza to lemmatize target terms.")

    args = parser.parse_args()

    all_line_testsets = []

    target_stanza_nlp = stanza.Pipeline(args.target_lang, processors='tokenize,pos,lemma,depparse')

    term_lemmas = {}

    with open(args.annotated_file,encoding="utf8") as annotated:
        for line in annotated:
            source,target = line.split('\t')
            source_term_spans = re.findall(r"(\b(\d).+\2\b)",source)
            target_term_spans = re.findall(r"(\b(\d).+\2\b)",target)
            term_pairs = zip(
                sorted(source_term_spans,key=lambda x: x[1]),
                sorted(target_term_spans,key=lambda x: x[1]))
            line_terms = [TermAndTranslations(source[0],target[0],source[1]) for (source,target) in term_pairs]

            #get all subsets of terms to generate the test variants
            line_testset = generate_all_test_pairs(line_terms,source,target)

            all_line_testsets.append(line_testset)

            if args.spm_output_file:
                #We need to analyze at least a single occurrence of each translation variant
                #to get the lemmas
                sents_for_analysis = []
                all_target_terms = list(itertools.chain.from_iterable([x.target_terms for x in line_terms]))
                for source,target in line_testset:
                    sent_added = False
                    if not all_target_terms:
                        break
                    for target_term in all_target_terms[:]:
                        if target_term in target:
                            clean_target = re.sub(r'<term[^>]+tgt="(?P<tgt_term>[^"]+)"[^>]*>(?P<src_term>[^<]+)</term>', r"\2", target)
                            if not sent_added:
                                sents_for_analysis.append(clean_target)
                                sent_added = True
                            all_target_terms.remove(target_term)
                stanza_input = "\n\n".join(sents_for_analysis)
                stanza_sents = target_stanza_nlp(stanza_input).sentences

                all_target_terms = list(itertools.chain.from_iterable([x.target_terms for x in line_terms]))

                for stanza_sent in stanza_sents:
                    for target_term in all_target_terms:
                        term_pos = stanza_sent.text.find(target_term)
                        if term_pos != -1:
                            lemma_start = [x for x in stanza_sent.words if x.start_char == term_pos]
                            lemma_end = [x for x in stanza_sent.words if x.end_char == term_pos+len(target_term)]




        spm_testsets = generate_spm_lines(
            all_line_testsets,
            args.spm_source_model,
            args.spm_target_model,
            args.term_start_tag,
            args.term_end_tag,
            args.trans_end_tag
        )