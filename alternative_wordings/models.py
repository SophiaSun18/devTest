import spacy
import difflib
from difflib import Differ, SequenceMatcher
from mbart_model import mbartAlt
from marian_model import marianAlt
import torch

torch.cuda.empty_cache()
nlp = spacy.load("en_core_web_sm")
mbart = mbartAlt("nl_XX")
print("here")
# marian = marianAlt(">>es<<")
use_mbart = True

# Dictionary to convert pronouns for passive to active voice
obj_to_subj_pronouns = {
    "her": "she",
    "him": "he",
    "whom": "who",
    "me": "I",
    "us": "we",
    "them": "they",
}

off_limits = []

# get prepositional phrases
# adapted from https://stackoverflow.com/questions/39100652/python-chunking-others-than-noun-phrases-e-g-prepositional-using-spacy-etc
def get_pps(doc):
    pps = []
    for token in doc:
        if token.pos_ == "ADP":
            pp = " ".join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
        if token.dep_ == "prep":
            off_limits.append(" ".join([tok.orth_ for tok in token.subtree]))
    return pps


def get_adv_clause(doc):
    clauses = []
    for token in doc:
        if (
            token.dep_ == "advcl"
            or token.dep_ == "npadvmod"
            or token.dep_ == "advmod"
            or token.pos_ == "SCONJ"
        ):
            clause = " ".join([tok.orth_ for tok in token.subtree])
            # fix apostrophy s issues
            clause = clause.replace(" '", "'")
            clauses.append(clause)
    return clauses


def capitalize_first_word(phrase):
    return phrase.split(" ")[0].capitalize() + " " + " ".join(phrase.split(" ")[1:])


def get_score(doc, sentence, results):
    # count content words in original and each alternative to catch options that repeat or leave off important phrases
    important_words = [
        token.text for token in doc if token.is_stop != True and token.is_punct != True
    ]
    wordcount = []
    for word in important_words:
        wordcount.append((word, sentence.count(word)))
    for resultset in results:
        idx = 0
        for score, sen in resultset:
            resdoc = nlp(sen)
            important = [
                token.text
                for token in resdoc
                if token.is_stop != True and token.is_punct != True
            ]
            # allow +2 for mbart
            if len(important) - len(important_words) not in [-1, 0, 1, 2]:
                resultset[idx] = (score - 10, sen)
            else:
                for el in wordcount:
                    if sen.count(el[0]) > el[1]:
                        resultset[idx] = (score - 10, sen)
            idx += 1
    return score


def get_color_chunks(all_sorted, doc, score):
    # select prepositional and noun phrases to be highlighted
    top = nlp(all_sorted[0][0][1])
    highlight = []
    for pphrase in get_pps(top):
        highlight.append(pphrase)
    for chunk in doc.noun_chunks:
        if chunk.text not in " ".join(highlight):
            highlight.append(chunk.text)
    color_code_chunks = []
    for optionset in all_sorted:
        color_code_subset = []
        for score, text in optionset:
            if score > -10:
                ph_and_idx = []
                for ph in highlight:
                    starting_idx = text.lower().find(ph.lower())
                    ph_and_idx.append((ph, starting_idx))

                order = sorted((score, text) for text, score in ph_and_idx)
                ordered_phrases = [phrase for score, phrase in order]

                new_sentence = text
                final_sentence = []
                x = 0
                for phrase in ordered_phrases:
                    if phrase.lower() in text.lower():
                        starting_idx = text.lower().find(phrase.lower())
                        final_sentence.append(
                            (new_sentence.lower().split(phrase.lower())[0], 0)
                        )
                        new_sentence = new_sentence.lower().split(phrase.lower())[-1]
                        final_sentence.append((phrase, highlight.index(phrase) + 1))
                    x += 1
                final_sentence.append((new_sentence, 0))
                color_code_subset.append(final_sentence)
        color_code_chunks.append(color_code_subset)

    # messy way to capitalize sentences
    for group in color_code_chunks:
        for chunk in group:
            if chunk[0][0] == "":
                first = chunk[1][0]
                capitalized = capitalize_first_word(first)
                chunk[1] = (capitalized, chunk[1][1])
            else:
                first = chunk[0][0]
                capitalized = capitalize_first_word(first)
                chunk[0] = (capitalized, chunk[0][1])
    return color_code_chunks


# summary: calculate_differences compares each alternative to an original sentence
#          a numerical representation of the differences is returned for each sentence
# parameters: alternatives, a list of alternative sentences to compare
#             original_sentence, sentence to compare against alternatives
# returns: list of list of differences (as integers) between each alternative and the original sentence
#######################################################################################
def calculate_differences(alternatives, original_sentence, prefix):
    differences = []
    for option in alternatives:
        diffs = []
        a = original_sentence.split()
        print(a)
        b = option.split()
        print(b)
        s = SequenceMatcher(None, a, b)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag != "equal":
                x = j1 - len(prefix.split()) + 1
                for string in b[j1:j2]:
                    print(string)
                    diffs.append(x)
                    x += 1
        differences.append(diffs)
    return differences


def get_phrases(doc):
    phrases = []
    for pphrase in get_pps(doc):
        # messy way to capitalize the first word without lowercasing the others
        phrases.append(capitalize_first_word(pphrase))

    # get subject after agent
    pronoun_to_convert = ""
    for sent in doc.sents:
        for token in sent:
            # Checks if there is a pronoun after agent for passive sentences
            if (
                token.pos_ == "PRON"
                and sent[(token.i - sent.start) - 1].dep_ == "agent"
            ):
                pronoun_to_convert = token.text

    # get noun chunks that aren't OPs
    for chunk in doc.noun_chunks:
        text = chunk.text
        valid = True
        for phr in off_limits:
            if text in phr:
                valid = False
        if valid:
            # Check if pronoun needs to be converted.
            if text == pronoun_to_convert:
                # Switch to correct pronoun
                text = obj_to_subj_pronouns.get(pronoun_to_convert)
            phrases.append(capitalize_first_word(text))

    # get adverbial modifiers and clauses
    for clause in get_adv_clause(doc):
        phrases.append(capitalize_first_word(clause))

    # get clause beginnings
    wordlist = [t.orth_ for t in doc]
    for token in doc:
        if token.dep_ == "nsubj":
            mystr = (
                " ".join([t.orth_ for t in token.lefts])
                + token.text
                + " "
                + token.head.orth_
            )

            phraselist = wordlist[
                wordlist.index(token.orth_) : wordlist.index(token.head.orth_) + 1
            ]
            phrases.append(" ".join(phraselist).capitalize())

            wordlist.remove(token.orth_)

    return phrases


def incremental_alternatives(sentence, prefix, recalculation):
    doc = nlp(sentence)
    highlight = []
    for chunk in doc.noun_chunks:
        highlight.append(chunk.text)
    new_sentence = sentence
    final_sentence = []
    for phrase in highlight:
        final_sentence.append((new_sentence.lower().split(phrase.lower())[0], 0))
        new_sentence = new_sentence.lower().split(phrase.lower())[-1]
        final_sentence.append((phrase, highlight.index(phrase) + 1))
    final_sentence.append((new_sentence, 0))
    return {"chunks": final_sentence}
    # return marian.incremental_alternatives(sentence, prefix, recalculation)


# summary: generate_alternatives generates alternative sentences for a given english sentence.
# parameters: english, the original sentence to get alternatives of
# returns: dict including:
#             alternatives, a list of lists of sentences with each outer list having a
#               different forced starting prefix and inner lists having different endings
#             color_coding, a list for each alternative sentence separating the sentence
#               into its sentence parts
#######################################################################################
def generate_alternatives(english):
    sentence = english
    doc = nlp(sentence)
    phrases = get_phrases(doc)

    results = []

    if use_mbart:
        results = mbart.get_prefix_alts(sentence, phrases)
    else:
        results = marian.get_prefix_alts(sentence, phrases)

    score = get_score(doc, sentence, results)

    # sort results with highest score first
    all_sorted = sorted(results, key=lambda x: x[0])[::-1]

    color_code_chunks = get_color_chunks(all_sorted, doc, score)

    alternatives = []
    scores = []
    for subset in all_sorted:
        altgroup = []
        for score, result in subset:
            altgroup.append(result)
        alternatives.append(altgroup)

    print(alternatives)

    return {"alternatives": alternatives, "colorCoding": color_code_chunks}


# summary: completion
# parameters: sentence, the sentence to generate alternatives of
#             prefix, A prefix to force in generating new sentence
# returns: dict including:
#               endings, list possible alternative sentence endings
#               differences, a list for each alternative sentence specifying the differences
#                   between it and the original
#######################################################################################
def completion(sentence, prefix):
    prefix = prefix.replace(" ", "", 1)
    top5 = marian.completion(sentence, prefix)
    # caculate difference in words for each alternative
    differences = calculate_differences(top5, sentence, prefix)
    print("prefix length: ", len(prefix.split()))

    endings = []
    for s in top5:
        endings.append(s.replace(prefix, ""))

    return {"endings": endings, "differences": differences}


def generate_constraints(sentence, constraints):
    print(sentence)
    new_constraints = []
    for idx, constraint in enumerate(constraints):
        # too simple filtering of "the"
        constraint = constraint.replace("the ", "")
        constraint = constraint.replace("The ", "")
        if idx != 0:
            # constraint = constraint[0].lower() + constraint[1:]
            pass
        new_constraints.append(constraint)
    # doc = nlp(sentence)
    # possible_prefixes = get_phrases(doc)
    # usable_prefix = ""
    # for prefix in possible_prefixes:
    #     if constraints[0] in prefix:
    #         usable_prefix = prefix
    #     else:
    #         if new_constraints[0] in prefix:
    #             usable_prefix = prefix
    #     print(usable_prefix)
    # print(usable_prefix)
    away = mbart.bart.translate(sentence)
    away = mbart.clean_lang_tok(away)
    resultset, word_alternatives = mbart.round_trip(away, new_constraints)
    return {"result": resultset[0][1], "word_alternatives": word_alternatives}


if __name__ == "__main__":
    # test for function output
    # genAltReturn = generate_alternatives(
    #     "The church currently maintains a program of ministry, outreach, and cultural events."
    # )
    # print("generate_alternatives()")
    # print(genAltReturn)

    # genincrReturn = marian.incremental_alternatives(
    #     "The church currently maintains a program of ministry, outreach, and cultural events.",
    #     "",
    #     False,
    # )
    # print("incremental_alternatives()")
    # print(genincrReturn)

    # completionReturn = completion(
    #     "The church currently maintains a program of ministry, outreach, and cultural events.",
    #     "The church presently",
    # )
    # print("completion()")
    # print(completionReturn)
    print(
        generate_constraints(
            "Yellowstone National Park was established by the US government in 1972 as the world's first legislated effort at nature conservation.",
            [
                "the US government",
                "Yellowstone National Park",
                "the world's first legislated effort",
                "nature conservation",
            ],
        )
    )
