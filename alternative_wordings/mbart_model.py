import torch
from fairseq.token_generation_constraints import pack_constraints
from fairseq.models.transformer import TransformerModel
import re

word_alts = False


class mbartAlt:
    def __init__(self, lang: str):
        self.bart = TransformerModel.from_pretrained(
            "mbart50.ft.nn",
            checkpoint_file="model.pt",
            data_name_or_path="mbart50.ft.nn",
            bpe="sentencepiece",
            sentencepiece_model="mbart50.ft.nn/sentence.bpe.model",
            lang_dict="mbart50.ft.nn/ML50_langs.txt",
            target_lang=lang,
            source_lang="en_XX",
            encoder_langtok="src",
        )
        self.bart.eval()
        self.bart.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.lang = lang

    def constraint2tensor(self, constraints: [str]):
        for i, constraint_list in enumerate(constraints):
            constraints[i] = [
                # encode with src_dict as this becomes tgt
                self.bart.src_dict.encode_line(
                    self.bart.apply_bpe(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]
        return pack_constraints(constraints)

    def clean_lang_tok(self, input: str):
        return re.sub("^[\[].*[\]] ", "", input)

    def sample(self, sentence, beam, verbose, **kwargs):
        tokenized_sentence = [self.bart.encode(sentence)]
        hypos = self.bart.generate(tokenized_sentence, beam, verbose, **kwargs)[0]
        word_alternatives = self.word_alternatives(
            torch.tensor(
                [self.bart.binarize(self.lang)[0].tolist()]
                + tokenized_sentence[0].tolist()
            ),
            hypos[0]["tokens"],
        )
        return hypos, word_alternatives

    def round_trip(self, sentence: str, constraints: [str]):
        print(constraints)
        constraints_tensor = self.constraint2tensor([constraints])
        # prefix = (
        #     self.bart.tgt_dict.encode_line(
        #         self.bart.apply_bpe(prefix),
        #         append_eos=False,
        #         add_if_not_exist=False,
        #     )
        #     .long()
        #     .unsqueeze(0)
        #     .to(self.bart._float_tensor.device)
        # )
        # switch translation direction
        orig_tgt = self.bart.task.args.target_lang
        orig_src = self.bart.task.args.source_lang
        self.bart.task.args.target_lang = orig_src
        self.bart.task.args.source_lang = orig_tgt

        returned, word_alternatives = self.sample(
            sentence,
            beam=100,
            verbose=True,
            constraints="ordered",
            inference_step_args={
                "constraints": constraints_tensor,
            },
            no_repeat_ngram_size=4,
            max_len_a=1,
            max_len_b=2,
            unkpen=10,
        )
        resultset = []
        for i in range(len(returned)):
            resultset.append(
                (
                    returned[i]["score"],
                    self.clean_lang_tok(self.bart.decode(returned[i]["tokens"])),
                )
            )
        # print(resultset)
        # restore original translation direction
        self.bart.task.args.target_lang = orig_tgt
        self.bart.task.args.source_lang = orig_src
        return resultset, word_alternatives

    def get_prefix_alts(self, sentence, prefixes: [str]):
        away = self.bart.translate(sentence)
        away = self.clean_lang_tok(away)
        return [self.round_trip(away, [prefix]) for prefix in prefixes]

    def word_alternatives(self, away_tokens, hypos_tokens):
        alternatives = []
        # _float_tensor.device is the way fairseq gets current device to use within their code
        # get language model scores
        lm_scores = self.bart.models[0](
            away_tokens.unsqueeze(0).to(self.bart._float_tensor.device),
            torch.tensor([len(away_tokens)]).to(self.bart._float_tensor.device),
            hypos_tokens[:-1].unsqueeze(0).to(self.bart._float_tensor.device),
        )[0][0]
        # do not compute sim score for language code
        sim_scores = self.similar_words(hypos_tokens[1:])
        # combine sim and lm scores
        for idx, word_scores in enumerate(lm_scores):
            alternatives.append(
                [
                    word.replace("\u2581", " ")
                    for word in self.bart.string(
                        (word_scores * 0.2 + torch.tensor(sim_scores[idx]) * 0.8)
                        .topk(10)
                        .indices
                    ).split(" ")
                ]
            )
        return alternatives

    def similar_words(self, word_tokens):
        sim_scores = []
        # get bart embedding
        emb = self.bart.models[0].decoder.output_projection.weight
        for token in word_tokens:
            # get similar words
            word_emb = emb[token]
            sim_scores.append(torch.matmul(emb, word_emb))
        return sim_scores


if __name__ == "__main__":
    mbart = mbartAlt("nl_XX")
    # print(torch.cuda.is_available())
    # print(
    #     mbart.get_prefix_alts(
    #         "She shot the cow during a time of scarcity to feed her hungry family.",
    #         [
    #             "During a time of scarcity",
    #             "Of scarcity",
    #             "She ",
    #             "The cow",
    #             "Her hungry family",
    #             "To feed her hungry family",
    #             "She shot",
    #         ],
    #     )
    # )
    # away = mbart.bart.translate(
    #     "Yellowstone National Park was established by the US government in 1972 as the world's first legislated effort at nature conservation."
    # )
    # away = mbart.clean_lang_tok(away)
    # print(
    #     mbart.round_trip(
    #         away,
    #         [
    #             "the world's first legislated effort",
    #             "nature conservation",
    #             "the US government",
    #             "Yellowstone National Park",
    #         ],
    #         "As the world's first legislated effort at nature conservation",
    #     )
    # )
    away = mbart.bart.translate(
        "Researchers found that heart attacks can be caused by stress."
    )
    away = mbart.clean_lang_tok(away)
    print(
        mbart.round_trip(
            away,
            ["Heart attacks", "caused", "stress", "researchers"],
        )
    )