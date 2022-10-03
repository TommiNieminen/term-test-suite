import argparse
import itertools
import regex
import sys

import sentencepiece as spm
from itertools import chain, combinations
import stanza

term_tag_regex = regex.compile(r'<term[^>]+tgt="(?P<tgt_term>[^"]+)"[^>]*>(?P<src_term>[^<]+)</term>')

class TermAndTranslations:

    def __init__(self,source_string,target_string,term_index):
        self.source_string = source_string
        self.target_string = target_string
        self.term_index = term_index
        self.source_term = source_string.replace(self.term_index,"").replace("¤","")
        self.target_terms = [x for x in target_string.replace("¤","").split(self.term_index) if x]

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
        trans_end_tag,
        term_lemma_dict):
    source_sp_model = spm.SentencePieceProcessor(spm_source_model)
    target_sp_model = spm.SentencePieceProcessor(spm_target_model)

    #the tag sp needs to be reverted, so store these
    sp_term_start_tag = " ".join(source_sp_model.encode_as_pieces(term_start_tag))
    sp_term_end_tag = " ".join(source_sp_model.encode_as_pieces(term_end_tag))


    all_testset_sources = []
    for line_testset in all_line_testsets:
        testset_sources = []
        for source,target in line_testset:
            terms = list(term_tag_regex.finditer(source))

            #first mark terms in the source with the term start and end tags and spm source
            tagged_source = source
            for term in terms:
                source_term = term.group("src_term").strip()
                tagged_source_term = f'{term_start_tag} {term.group("src_term")} {term_end_tag}'
                tagged_source = tagged_source.replace(term.group(0),tagged_source_term)
            sp_source = " ".join(source_sp_model.encode_as_pieces(tagged_source))
            #revert tag sp
            sp_source = sp_source.replace(sp_term_start_tag,term_start_tag)
            sp_source = sp_source.replace(sp_term_end_tag, term_end_tag)

            #then lemmatize target terms and append them to source terms
            for term in terms:
                source_term = term.group("src_term").strip()
                sp_source_term = " ".join(source_sp_model.encode_as_pieces(source_term))
                target_term = term.group("tgt_term").strip()
                term_lemma = term_lemma_dict[target_term]
                sp_term_lemma = " ".join(target_sp_model.encode_as_pieces(term_lemma))
                tagged_sp_source_term = f'{term_start_tag} {sp_source_term} {term_end_tag}'
                soft_constraint = f'{tagged_sp_source_term} {sp_term_lemma} {trans_end_tag}'
                sp_source = sp_source.replace(tagged_sp_source_term,soft_constraint)

            target_without_terms = term_tag_regex.sub("",target)
            sp_target = " ".join(target_sp_model.encode_as_pieces(target_without_terms))

            testset_sources.append((sp_source,sp_target))
        all_testset_sources.append(testset_sources)
    return all_testset_sources

def get_bare_stanza_lemma(stanza_token):
    # Hash symbol is used as compound divider in the lemmas, problem is that
    # if the compound was hyphenated, the hash overwrites the hyphen, so restore hyphen
    # note that this will probably fail in some cases, where there are more than two parts to the
    # compound and some are hyphenated and others not (but the effect will be small).
    if stanza_token and stanza_token.lemma and stanza_token.text:
        if "#" in stanza_token.lemma and "#" not in stanza_token.text:
            if "-" in stanza_token.text:
                return stanza_token.lemma.replace('#','-')
            else:
                return stanza_token.lemma.replace('#','')
        else:
            return stanza_token.lemma
    else:
        sys.stderr.write(f"Invalid stanza token {stanza_token}\n")
        return None

def add_line_term_lemmas_to_dict(line_terms,line_testset,target_stanza_nlp,term_lemma_dict):
    # We need to analyze at least a single occurrence of each translation variant
    # to get the lemmas
    sents_for_analysis = []
    all_target_terms = list(itertools.chain.from_iterable([x.target_terms for x in line_terms]))
    for source, target in line_testset:
        sent_added = False
        if not all_target_terms:
            break
        for target_term in all_target_terms[:]:
            if target_term in term_lemma_dict:
                all_target_terms.remove(target_term)
                continue
            if target_term in target:
                clean_target = regex.sub(r'<term[^>]+tgt="(?P<tgt_term>[^"]+)"[^>]*>(?P<src_term>[^<]+)</term>', r"\2",
                                      target)
                if not sent_added:
                    sents_for_analysis.append(clean_target)
                    sent_added = True
                all_target_terms.remove(target_term)
    stanza_input = "\n\n".join(sents_for_analysis)
    stanza_sents = target_stanza_nlp(stanza_input).sentences

    all_target_terms = list(itertools.chain.from_iterable([x.target_terms for x in line_terms]))

    for stanza_sent in stanza_sents:
        for target_term in all_target_terms:
            if target_term in term_lemma_dict:
                continue
            term_pos_in_sent = stanza_sent.text.find(target_term)
            if term_pos_in_sent != -1:
                term_pos = term_pos_in_sent+stanza_sent.words[0].start_char
                lemma_start = [x for x in stanza_sent.words if x.start_char == term_pos][0]
                lemma_end = [x for x in stanza_sent.words if x.end_char == term_pos + len(target_term)][0]
                term_words = []
                for id in range(lemma_start.id, lemma_end.id + 1):
                    term_words.append([x for x in stanza_sent.words if x.id == id][0])
                term_lemma_string = " ".join([get_bare_stanza_lemma(x) for x in term_words])
                term_lemma_dict[target_term] = term_lemma_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a terminology translation test suite from manually annotated files.")
    parser.add_argument("--annotated_file", type=str,
                        help="Tab separated file (source TAB translation) containing term annotations " +
                        "where terms are indicated with pre- and post-fixed ids. Example: source: source" +
                        "with 1term1 or 2another term2 TAB translation with 1term1 or 2another term2" +
                        "Target side can contain alternatives, e.g 1first alt1second alt1")
    parser.add_argument("--sp_source", type=str,
                        help="Output source file that has been segmented with the provided source spm model, " +
                        "with terms added using the term tags from the spm model and segmented with the " +
                        "target spm model")
    parser.add_argument("--sp_target", type=str,
                        help="Output target file that has been segmented with the provided target spm model. " +
                        "This file can be used for scoring translations with different soft constraints.")
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

    if args.sp_source:
        target_stanza_nlp = stanza.Pipeline(args.target_lang, processors='tokenize,pos,lemma,depparse')
        term_lemma_dict = {}

    term_lemmas = {}

    with open(args.annotated_file,encoding="utf8") as annotated:
        for line in annotated:
            source,target = line.split('\t')
            #TODO: change the encoding scheme to deal with e.g. 1st
            source_term_spans = regex.findall(r"(¤(\d).+\2¤)",source)
            target_term_spans = regex.findall(r"(¤(\d).+\2¤)",target)
            term_pairs = zip(
                sorted(source_term_spans,key=lambda x: x[1]),
                sorted(target_term_spans,key=lambda x: x[1]))
            line_terms = [TermAndTranslations(source[0],target[0],source[1]) for (source,target) in term_pairs]

            #get all subsets of terms to generate the test variants
            line_testset = generate_all_test_pairs(line_terms,source,target)

            all_line_testsets.append(line_testset)

            if args.sp_source:
                add_line_term_lemmas_to_dict(line_terms, line_testset, target_stanza_nlp,term_lemma_dict)




    spm_testsets = generate_spm_lines(
        all_line_testsets,
        args.spm_source_model,
        args.spm_target_model,
        args.term_start_tag,
        args.term_end_tag,
        args.trans_end_tag,
        term_lemma_dict
    )

    with \
        open(args.sp_source,'wt',encoding="utf8") as sp_source, \
        open(args.sp_target,'wt', encoding="utf8") as sp_target:
        for spm_testset in spm_testsets:
            for source,target in spm_testset:
                sp_source.write(source + '\n')
                sp_target.write(target + '\n')
