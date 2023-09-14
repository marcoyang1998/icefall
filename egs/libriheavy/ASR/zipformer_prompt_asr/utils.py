from typing import Dict, List, Tuple, TextIO, Union, Iterable
import ast
from collections import defaultdict
from lhotse import load_manifest, load_manifest_lazy
from lhotse.cut import Cut, CutSet
from text_normalization import remove_non_alphabetic
from tqdm import tqdm
import os

import kaldialign
import logging

def get_facebook_biasing_list(
    test_set: str,
    use_distractors: bool = False,
    num_distractors: int = 100,
) -> Dict:
    # Get the biasing list from the meta paper: https://arxiv.org/pdf/2104.02194.pdf
    assert num_distractors in (0, 100,500,1000,2000), num_distractors
    if num_distractors == 0:
        if test_set == "test-clean":
            biasing_file = f"data/context_biasing/fbai-speech/is21_deep_bias/ref/test-clean.biasing_100.tsv"
        elif test_set == "test-other":
            biasing_file = f"data/context_biasing/fbai-speech/is21_deep_bias/ref/test-other.biasing_100.tsv"
        else:
            raise ValueError(f"Unseen test set {test_set}")
    else:
        if test_set == "test-clean":
            biasing_file = f"data/context_biasing/fbai-speech/is21_deep_bias/ref/test-clean.biasing_{num_distractors}.tsv"
        elif test_set == "test-other":
            biasing_file = f"data/context_biasing/fbai-speech/is21_deep_bias/ref/test-other.biasing_{num_distractors}.tsv"
        else:
            raise ValueError(f"Unseen test set {test_set}")
    
    f = open(biasing_file, 'r')
    data = f.readlines()
    f.close()
    
    output = dict()
    for line in data:
        id, _, l1, l2 = line.split('\t')
        if use_distractors:
            biasing_list = ast.literal_eval(l2)
        else:
            biasing_list = ast.literal_eval(l1)
        biasing_list = [w.strip().upper() for w in biasing_list] 
        output[id] = " ".join(biasing_list)
         
    return output

def brian_biasing_list(level: str):
    # The biasing list from Brian's paper: https://arxiv.org/pdf/2109.00627.pdf
    import glob
    root_dir = f"data/context_biasing/LibriSpeechBiasingLists/{level}Level"
    all_files = glob.glob(root_dir + "/*")
    biasing_dict = {}
    for f in all_files:
        k = f.split('/')[-1]
        fin = open(f, 'r')
        data = fin.read().strip().split()
        biasing_dict[k] = " ".join(data)
        fin.close()
        
    return biasing_dict

def get_rare_words(subset: str, min_count: int):
    """Get a list of rare words appearing less than `min_count` times

    Args:
        subset: 
        min_count (int): Count of appearance
    """
    txt_path = f"data/tmp/transcript_words_{subset}.txt"
    rare_word_file = f"data/context_biasing/{subset}_rare_words_{min_count}.txt"
    
    if os.path.exists(rare_word_file):
        print("File exists, do not proceed!")
        return
    print(f"Finding rare words in the manifest.")
    count_file = f"data/tmp/transcript_words_{subset}_count.txt"
    if not os.path.exists(count_file):
        with open(txt_path, "r") as file:
            words = file.read().upper().split()
            word_count = {}
            for word in words:
                word = remove_non_alphabetic(word, strict=False)
                word = word.split()
                for w in word:
                    if w not in word_count:
                        word_count[w] = 1
                    else:
                        word_count[w] += 1
        
        with open(count_file, 'w') as fout:
            for w in word_count:
                fout.write(f"{w}\t{word_count[w]}\n")
    else:
        word_count = {}
        with open(count_file, 'r') as fin:
            word_count = fin.read().strip().split('\n')
            word_count = [pair.split('\t') for pair in word_count]
            word_count = sorted(word_count, key=lambda w: int(w[1]), reverse=True)

    print(f"A total of {len(word_count)} words appeared!")
    rare_words = []
    for k in word_count:
        if int(word_count[k]) <= min_count:
            rare_words.append(k+"\n")
    print(f"A total of {len(rare_words)} appeared <= {min_count} times")

    with open(rare_word_file, 'w') as f:
        f.writelines(rare_words)
    
def add_context_list_to_manifest(subset: str, min_count: int):
    """Generate a context list of rare words for each utterance in the manifest

    Args:
        subset (str): Subset
        min_count (int): The min appearances

    """
    rare_words_file = f"data/context_biasing/{subset}_rare_words_{min_count}.txt"
    manifest_dir = f"data/fbank/libriheavy_cuts_{subset}.jsonl.gz"
    
    target_manifest_dir = manifest_dir.replace(".jsonl.gz", f"_with_context_list_min_count_{min_count}.jsonl.gz")
    if os.path.exists(target_manifest_dir):
        print(f"Target file exits at {target_manifest_dir}!")
        return
    
    print(f"Reading rare words from {rare_words_file}")
    with open(rare_words_file, "r") as f:
        rare_words = f.read()
    rare_words = rare_words.split("\n")
    rare_words = set(rare_words)
    print(f"A total of {len(rare_words)} rare words!")
    
    cuts = load_manifest_lazy(manifest_dir)
    print(f"Loaded manifest from {manifest_dir}")
    
    def _add_context(c: Cut):
        splits = remove_non_alphabetic(c.supervisions[0].texts[0], strict=False).upper().split()
        found = []
        for w in splits:
            if w in rare_words:
                found.append(w)
        c.supervisions[0].context_list = " ".join(found)
        return c

    cuts = cuts.map(_add_context)
        
    cuts.to_file(target_manifest_dir)
    print(f"Saved manifest with context list to {target_manifest_dir}")


def check(subset: str, min_count: int):
    # Used to show how many samples in the training set have a context list
    print("Calculating the stats over the manifest")
    manifest_dir = f"data/fbank/libriheavy_cuts_{subset}_with_context_list_min_count_{min_count}.jsonl.gz"
    cuts = load_manifest_lazy(manifest_dir)
    total_cuts = len(cuts)
    has_context_list = [c.supervisions[0].context_list != "" for c in cuts]
    context_list_len = [len(c.supervisions[0].context_list.split()) for c in cuts]
    print(f"{sum(has_context_list)}/{total_cuts} cuts have context list! ")
    print(f"Average length of non-empty context list is {sum(context_list_len)/sum(has_context_list)}")


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
    compute_CER: bool = False,
    biasing_words: List[str] = None,
) -> float:
    """Write statistics based on predicted results and reference transcripts. It also calculates the
    biasing word error rate as described in https://arxiv.org/pdf/2104.02194.pdf

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cut_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
      biasing_words:
        All the words in the biasing list
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"
    
    if compute_CER:
        for i, res in enumerate(results):
            cut_id, ref, hyp = res
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results[i] = (cut_id, ref, hyp)
    
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        for ref_word, hyp_word in ali:
            if ref_word == ERR: # INSERTION
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR: # DELETION
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word: # SUBSTITUTION
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word
                    if ref_word == hyp_word
                    else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted(
        [(v, k) for k, v in subs.items()], reverse=True
    ):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    unbiased_word_counts = 0
    unbiased_word_errs = 0
    biased_word_counts = 0
    biased_word_errs = 0
    
    print("", file=f)
    print(
        "PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f
    )

    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        # number of appearances of "word" in reference text
        ref_count = corr + ref_sub + dels # correct + in ref but got substituted + deleted
        # number of appearances of "word" in hyp text
        hyp_count = corr + hyp_sub + ins
        
        
        if biasing_words is not None:
            if word in biasing_words:
                biased_word_counts += ref_count
                biased_word_errs += (ins + dels + ref_sub)
            else:
                unbiased_word_counts += ref_count
                unbiased_word_errs += (ins + dels + hyp_sub)

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    
    if biasing_words is not None:
        B_WER = "%.2f" % (100 *biased_word_errs/biased_word_counts)
        U_WER = "%.2f" % (100 *unbiased_word_errs/unbiased_word_counts)
        logging.info(f"Biased WER: {B_WER} [{biased_word_errs}/{biased_word_counts}] ")
        logging.info(f"Un-biased WER: {U_WER} [{unbiased_word_errs}/{unbiased_word_counts}]")
            

    return float(tot_err_rate)



if __name__=="__main__":
    #test_set = "test-clean"
    #get_facebook_biasing_list(test_set)
    subset = "medium"
    min_count = 460
    #get_rare_words(subset, min_count)
    add_context_list_to_manifest(subset=subset, min_count=min_count)
    check(subset=subset, min_count=min_count)