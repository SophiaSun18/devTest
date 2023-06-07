import torch
from transformers import MarianMTModel, MarianTokenizer


class CustomMTModel(MarianMTModel):
    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if not self.original_postprocess:
            if 0 < cur_len <= len(self.selected_tokens):
                force_token_id = self.selected_tokens[cur_len - 1]
                logits[
                    :, [x for x in range(logits.shape[1]) if x != force_token_id]
                ] = -float("inf")

        return MarianMTModel.adjust_logits_during_generation(
            self, logits, cur_len, max_length
        )


class marianAlt:
    def __init__(self, lang: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        en_ROMANCE_model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        self.en_ROMANCE_tokenizer = MarianTokenizer.from_pretrained(
            en_ROMANCE_model_name
        )
        self.en_ROMANCE = MarianMTModel.from_pretrained(en_ROMANCE_model_name).to(
            self.device
        )

        ROMANCE_en_model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
        self.ROMANCE_en_tokenizer = MarianTokenizer.from_pretrained(
            ROMANCE_en_model_name
        )
        self.ROMANCE_en = MarianMTModel.from_pretrained(ROMANCE_en_model_name).to(
            self.device
        )
        self.ROMANCE_en.__class__ = CustomMTModel

        self.lang = lang

    def translate(self, text, num_outputs):
        """Use beam search to get a reasonable translation of 'text'"""
        # Tokenize the source text
        self.ROMANCE_en_tokenizer.current_spm = (
            self.ROMANCE_en_tokenizer.spm_source
        )  # HACK!
        batch = self.ROMANCE_en_tokenizer(text, return_tensors="pt", padding=True).to(
            self.ROMANCE_en.device
        )
        # Run model
        num_beams = num_outputs
        translated = self.ROMANCE_en.generate(
            **batch,
            num_beams=num_beams,
            num_return_sequences=num_outputs,
            max_length=40,
            no_repeat_ngram_size=5
        )

        # Untokenize the output text.
        self.ROMANCE_en_tokenizer.current_spm = self.ROMANCE_en_tokenizer.spm_target
        return [
            self.ROMANCE_en_tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for t in translated
        ]

    # summary: Incremental_generation is used to generate alternative probable words for each word in a sentence
    # parameters: machine_translation, the spanish translation
    #             start, the forced beginning of the english.
    #             prefix_only, if true no new tokens will be generated after param 'start'
    # returns:the final text (will be the same as 'start' if prefix_only)
    #         the expected result (machine translation to english of the spanish input)
    #         list of tokens in the final sequence
    #         list of top 10 predictions for each token
    #         score for average predictability
    def incremental_generation(self, machine_translation, start, prefix_only):
        tokenizer = self.ROMANCE_en_tokenizer
        model = self.ROMANCE_en
        tokenized_prefix = tokenizer.convert_tokens_to_ids(
            self.en_ROMANCE_tokenizer.tokenize(start.strip())
        )
        prefix = torch.LongTensor(tokenized_prefix).to(self.device)

        batch = tokenizer(
            machine_translation.replace("<pad> ", ""), return_tensors="pt", padding=True
        ).to(self.device)
        original_encoded = model.get_encoder()(**batch)
        decoder_start_token = model.config.decoder_start_token_id
        partial_decode = (
            torch.LongTensor([decoder_start_token]).to(self.device).unsqueeze(0)
        )
        past = None

        # machine translation for comparative purposes
        translation_tokens = model.generate(**batch)
        auto_translation = tokenizer.decode(translation_tokens[0]).split("<pad>")[1]

        num_tokens_generated = 0
        prediction_list = []
        MAX_LENGTH = 100
        total = 0

        # generate tokens incrementally
        while True:
            model_inputs = model.prepare_inputs_for_generation(
                partial_decode,
                past=past,
                encoder_outputs=original_encoded,
                attention_mask=batch["attention_mask"],
                use_cache=model.config.use_cache,
            )
            with torch.no_grad():
                model_outputs = model(**model_inputs)

            next_token_logits = model_outputs[0][:, -1, :]
            past = model_outputs[1]

            # start with designated beginning
            if num_tokens_generated < len(prefix):
                next_token_to_add = prefix[num_tokens_generated]
            elif prefix_only == True:
                break
            else:
                next_token_to_add = next_token_logits[0].argmax()
                # stop adding when </s> is reached
                if next_token_to_add.item() == 0:
                    break

            # calculate score
            next_token_logprobs = next_token_logits - next_token_logits.logsumexp(
                1, True
            )
            token_score = next_token_logprobs[0][next_token_to_add].item()
            total += token_score

            # append top 10 predictions for each token to list
            decoded_predictions = []
            for tok in next_token_logits[0].topk(10).indices:
                decoded_predictions.append(
                    tokenizer.convert_ids_to_tokens(tok.item()).replace(
                        "\u2581", "\u00a0"
                    )
                )

            # list of lists of predictions
            prediction_list.append(decoded_predictions)

            # add new token to tokens so far
            partial_decode = torch.cat(
                (partial_decode, next_token_to_add.unsqueeze(0).unsqueeze(0)), -1
            )
            num_tokens_generated += 1

            # stop generating when max num tokens exceded
            if not (num_tokens_generated < MAX_LENGTH):
                break

        # list of tokens used to display sentence
        decoded_tokens = [
            sub.replace("\u2581", "\u00a0")
            for sub in tokenizer.convert_ids_to_tokens(partial_decode[0])
        ]
        decoded_tokens.remove("<pad>")

        final = tokenizer.decode(partial_decode[0]).replace("<pad>", "")
        score = round(total / (len(decoded_tokens)), 3)

        return {
            "final": final.lstrip(),
            "expected": auto_translation,
            "tokens": decoded_tokens,
            "predictions": prediction_list,
            "score": score,
        }

    # summary: incremental_alternatives is mainly used to generate the translation of the original sentence
    #          before feeding it to incremental_generation()
    # parameters: sentence, the sentence to generate alternatives of
    #             prefix, never used unless recalculation is true
    #             recalculation, if true the prefix is used to generate alternatives
    # returns: dict including:
    #               the final text
    #               the expected result (machine translation to english of the spanish input)
    #               list of tokens in the final sequence
    #               list of top 10 predictions for each token
    #               score for average predictability
    #######################################################################################
    def incremental_alternatives(self, sentence, prefix, recalculation):
        self.ROMANCE_en.original_postprocess = True
        english = self.lang + sentence
        eng_to_spanish = self.en_ROMANCE.generate(
            **self.en_ROMANCE_tokenizer(english, return_tensors="pt", padding=True).to(
                self.device
            )
        ).to(self.device)
        machine_translation = self.en_ROMANCE_tokenizer.decode(
            eng_to_spanish[0]
        ).replace("<pad> ", "")
        if recalculation:
            sentence = prefix
        return self.incremental_generation(machine_translation, sentence, False)

    def get_prefix_alts(self, sentence, phrases: [str]):
        # prepare input for translation
        self.ROMANCE_en.original_postprocess = True
        # Specifies target language to translate
        english = self.lang + sentence
        eng_to_spanish = self.en_ROMANCE.generate(
            **self.en_ROMANCE_tokenizer(english, return_tensors="pt", padding=True).to(
                self.device
            )
        ).to(self.device)
        machine_translation = self.en_ROMANCE_tokenizer.decode(
            eng_to_spanish[0]
        ).replace("<pad> ", "")

        results = []
        # generate alternatives starting with each selected phrase
        for selection in set(phrases):
            resultset = []
            self.ROMANCE_en_tokenizer.current_spm = self.ROMANCE_en_tokenizer.spm_target
            tokens = self.ROMANCE_en_tokenizer.tokenize(selection)
            self.ROMANCE_en.selected_tokens = (
                self.ROMANCE_en_tokenizer.convert_tokens_to_ids(tokens)
            )

            self.ROMANCE_en.original_postprocess = False
            print(self.ROMANCE_en.__class__)
            top50 = self.translate(">>en<<" + machine_translation, 50)
            for element in top50[0:3]:
                res = self.incremental_generation(
                    machine_translation, element, prefix_only=False
                )
                resultset.append((res["score"], res["final"]))
            results.append(resultset)
        return results

    # summary: completion
    # parameters: sentence, the sentence to generate alternatives of
    #             prefix, A prefix to force in generating new sentence
    # returns: dict including:
    #               endings, list possible alternative sentence endings
    #               differences, a list for each alternative sentence specifying the differences
    #                   between it and the original
    #######################################################################################
    def completion(self, sentence, prefix):
        self.ROMANCE_en.original_postprocess = True
        english = self.lang + sentence
        eng_to_spanish = self.en_ROMANCE.generate(
            **self.en_ROMANCE_tokenizer(english, return_tensors="pt", padding=True).to(
                self.device
            )
        ).to(self.device)
        machine_translation = self.en_ROMANCE_tokenizer.decode(
            eng_to_spanish[0]
        ).replace("<pad> ", "")

        self.ROMANCE_en_tokenizer.current_spm = self.ROMANCE_en_tokenizer.spm_target
        tokens = self.ROMANCE_en_tokenizer.tokenize(prefix)
        # add prefix to selected_tokens in order to force it in generation
        self.ROMANCE_en.selected_tokens = (
            self.ROMANCE_en_tokenizer.convert_tokens_to_ids(tokens)
        )

        self.ROMANCE_en.original_postprocess = False
        top5 = self.translate(">>en<<" + machine_translation, 5)
        return top5


if __name__ == "__main__":
    marian = marianAlt(">>es<<")
    print(
        marian.get_prefix_alts(
            "She shot the cow during a time of scarcity to feed her hungry family.",
            [
                "During a time of scarcity",
                "Of scarcity",
                "She ",
                "The cow",
                "Her hungry family",
                "To feed her hungry family",
                "She shot",
            ],
        )
    )
